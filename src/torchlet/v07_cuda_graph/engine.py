import asyncio
import torch
from torch import Tensor

from .forward_params import ForwardParams
from .request import Request
from .kvcache import KVCache
from .scheduler import Scheduler


def _config_get(config: dict, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


class Engine:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        stop_ids,
        max_decode_slots: int = 4,
        dummy_token_id: int = 0,
    ):
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("v07_cuda_graph requires CUDA")

        config = model.config
        hidden_size = _config_get(config, "hidden_size")
        num_heads = _config_get(config, "num_attention_heads")
        num_kv_heads = _config_get(config, "num_key_value_heads", num_heads)
        max_seq_len = _config_get(config, "max_position_embeddings")

        self.input_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.scheduler = Scheduler(
            self.input_queue,
            device=device,
            max_decode_slots=max_decode_slots,
            dummy_token_id=dummy_token_id,
            max_seq_len=max_seq_len,
        )
        self.kvcache = KVCache(
            max_layer=_config_get(config, "num_hidden_layers"),
            max_decode_slots=max_decode_slots,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=hidden_size // num_heads,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stop_ids = stop_ids
        self.task: asyncio.Task | None = None

        self.decode_cuda_graph_ready = False
        with torch.cuda.device(self.device):
            self.decode_cuda_graph = torch.cuda.CUDAGraph()

    def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.step_loop())

    async def shutdown(self):
        if self.task is None:
            return

        task = self.task
        self.task = None
        if task.done():
            return

        task.cancel()
        done_reqs = self.scheduler.fail_all(asyncio.CancelledError())
        self.complete_reqs(done_reqs)
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def add_requests(self, req_batch: list[Request]):
        await self.scheduler.add_requests(req_batch)

    async def step(self):
        prefill_reqs, decode_reqs = await self.scheduler.schedule()
        assert prefill_reqs or decode_reqs

        with torch.no_grad():
            if prefill_reqs:
                model_input = self.scheduler.build_model_input(
                    prefill_reqs, self.device
                )

                forward_params = ForwardParams(
                    req_indptr_cpu=model_input.req_indptr_cpu,
                    position_index=model_input.position_index,
                    slot_ids_cpu=model_input.slot_ids_cpu,
                    active_mask=None,
                    is_prefill=prefill_reqs.is_prefill,
                    kvcache=self.kvcache,
                )

                logits = self.model.forward(
                    model_input.flat_input_ids,
                    forward_params,
                    last_token_only=True,
                )  # [batch, vocab_size]
                gen_tok_id = self.sample(logits)  # [batch]
                self.scheduler.process_output(prefill_reqs, gen_tok_id, self.stop_ids)

            if decode_reqs:
                if not self.decode_cuda_graph_ready:
                    self._capture_decode_graph(decode_reqs)
                else:
                    self._run_decode_graph(decode_reqs)

        return True

    def _capture_decode_graph(self, decode_reqs):
        model_input = self.scheduler.build_model_input_decode(decode_reqs)

        self.decode_forward_params = ForwardParams(
            req_indptr_cpu=model_input.req_indptr_cpu,
            position_index=model_input.position_index,
            slot_ids_cpu=model_input.slot_ids_cpu,
            active_mask=model_input.active_mask,
            is_prefill=decode_reqs.is_prefill,
            kvcache=self.kvcache,
        )
        self.decode_flat_input_ids = model_input.flat_input_ids

        with torch.cuda.device(self.device):
            warmup_stream = torch.cuda.Stream(device=self.device)
            current_stream = torch.cuda.current_stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    logits = self.model.forward(
                        self.decode_flat_input_ids,
                        self.decode_forward_params,
                        last_token_only=True,
                    )
                    self.static_gen_tok_id = self.sample(logits)
            current_stream.wait_stream(warmup_stream)

            with torch.cuda.graph(self.decode_cuda_graph):
                logits = self.model.forward(
                    self.decode_flat_input_ids,
                    self.decode_forward_params,
                    last_token_only=True,
                )  # [batch, vocab_size]
                self.static_gen_tok_id = self.sample(logits)  # [batch]
        self.decode_cuda_graph_ready = True

        self.scheduler.process_output_decode(
            decode_reqs, self.static_gen_tok_id, self.stop_ids
        )

    def _run_decode_graph(self, decode_reqs):
        self.scheduler.build_model_input_decode(decode_reqs)
        with torch.cuda.device(self.device):
            self.decode_cuda_graph.replay()

        self.scheduler.process_output_decode(
            decode_reqs, self.static_gen_tok_id, self.stop_ids
        )

    def sample(self, logits: Tensor):
        return logits.argmax(dim=-1)

    def complete_reqs(self, requests: list[Request]):
        normal_reqs = [req for req in requests if req.error is None]
        if normal_reqs:
            out_tok_ids = [req.output_tokens for req in normal_reqs]
            output_strs = self.tokenizer.batch_decode(
                out_tok_ids, skip_special_tokens=True
            )
            assert len(normal_reqs) == len(output_strs)
            for req, out in zip(normal_reqs, output_strs):
                req.output = out

        for req in requests:
            req.done.set()

    async def step_loop(self):
        try:
            while True:
                if not await self.step():
                    return
                finished = self.scheduler.drain_finished_reqs()
                if finished:
                    self.complete_reqs(finished)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            done_reqs = self.scheduler.fail_all(exc)
            self.complete_reqs(done_reqs)
