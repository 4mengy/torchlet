import asyncio
import torch
from torch import Tensor

from .forward_params import ForwardParams
from .request import Request
from .kvcache import KVCache
from .scheduler import Scheduler


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
        self.input_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.scheduler = Scheduler(
            self.input_queue,
            device=device,
            max_decode_slots=max_decode_slots,
            dummy_token_id=dummy_token_id,
        )
        self.kvcache = KVCache()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stop_ids = stop_ids
        self.task: asyncio.Task | None = None

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
                    req_ids=model_input.req_ids,
                    req_indptr_cpu=model_input.req_indptr_cpu,
                    position_index=model_input.position_index,
                    active_mask_cpu=None,
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
                model_input = self.scheduler.build_model_input_decode(
                    decode_reqs, self.device
                )

                forward_params = ForwardParams(
                    req_ids=model_input.req_ids,
                    req_indptr_cpu=model_input.req_indptr_cpu,
                    position_index=model_input.position_index,
                    active_mask_cpu=model_input.active_mask_cpu,
                    is_prefill=decode_reqs.is_prefill,
                    kvcache=self.kvcache,
                )

                logits = self.model.forward(
                    model_input.flat_input_ids,
                    forward_params,
                    last_token_only=True,
                )  # [batch, vocab_size]
                gen_tok_id = self.sample(logits)  # [batch]
                self.scheduler.process_output_decode(
                    decode_reqs, gen_tok_id, self.stop_ids
                )
        return True

    def sample(self, logits: Tensor):
        return logits.argmax(dim=-1)

    def drain_finished_reqs(self):
        finished = self.scheduler.drain_finished_reqs()
        for req in finished:
            self.kvcache.remove(req.req_id)
        return finished

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
                finished = self.drain_finished_reqs()
                if finished:
                    self.complete_reqs(finished)
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            done_reqs = self.scheduler.fail_all(exc)
            for req in done_reqs:
                self.kvcache.remove(req.req_id)
            self.complete_reqs(done_reqs)
