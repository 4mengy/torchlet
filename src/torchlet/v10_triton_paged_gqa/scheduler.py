import asyncio
from .request import Request, RequestState
from dataclasses import dataclass
from torch import Tensor
import torch

from .kvcache import KVCache


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


@dataclass
class ScheduleReqs:
    requests: list[Request]
    is_prefill: bool


@dataclass
class ModelInput:
    flat_input_ids: Tensor
    req_indptr_cpu: Tensor
    position_index: Tensor
    slot_ids_cpu: None | list[int]
    active_mask: None | Tensor
    block_table: Tensor


class Scheduler:
    def __init__(
        self,
        queue: asyncio.Queue[Request],
        device,
        kvcache: KVCache,
        max_decode_slots: int = 4,
        dummy_token_id: int = 0,
    ):
        if max_decode_slots <= 0:
            raise ValueError("max_decode_slots must be positive")
        self.waiting = list()
        self.running = list()
        self.finished = list()
        self.queue = queue
        self.kvcache = kvcache
        self.kv_block_size = kvcache.block_size
        self.max_decode_slots = max_decode_slots
        self.dummy_token_id = dummy_token_id
        self.max_seq_len = kvcache.max_seq_len

        self.req_to_slot = dict()
        self.free_slots = [i for i in range(0, self.max_decode_slots)]

        max_blocks_per_req = _ceil_div(self.max_seq_len, self.kv_block_size)

        # Static block table reused by decode and CUDA Graph replay.
        # Shape: [max_decode_slots, max_blocks_per_req]
        self.decode_block_table = torch.full(
            (max_decode_slots, max_blocks_per_req),
            0,
            dtype=torch.long,
            device=device,
        )

        self.decode_input = torch.zeros(
            self.max_decode_slots, dtype=torch.long, device=device
        )
        self.decode_req_indptr_cpu = torch.arange(
            self.max_decode_slots + 1, dtype=torch.long
        )
        self.decode_position_index = torch.zeros(
            self.max_decode_slots, dtype=torch.long, device=device
        )
        self.decode_active_mask = torch.zeros(
            self.max_decode_slots, dtype=torch.bool, device=device
        )

    async def add_requests(self, requests: list[Request]):
        for req in requests:
            self._validate_request(req)
        for req in requests:
            await self.queue.put(req)

    def _validate_request(self, req: Request):
        if req.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.max_seq_len is None:
            return

        required_positions = len(req.input_tokens) + req.max_new_tokens - 1
        if required_positions > self.max_seq_len:
            raise ValueError(
                f"request {req.req_id} requires {required_positions} cache positions, "
                f"but max_seq_len is {self.max_seq_len}"
            )

    async def schedule(self) -> tuple[ScheduleReqs | None, ScheduleReqs | None]:
        decode_reqs = None
        if self.running:
            new_running = []
            for req in self.running:
                if self.kvcache.allocate(
                    req, _ceil_div(req.computed_tokens + 1, self.kv_block_size)
                ):
                    new_running.append(req)
                else:
                    req.state = RequestState.WAITING
                    self._release_slot(req)
                    self.waiting.insert(0, req)
            self.running = new_running
            if self.running:
                decode_reqs = ScheduleReqs(
                    requests=list(self.running),
                    is_prefill=False,
                )

        prefill_reqs = None
        waiting_reqs = []
        if decode_reqs is None and not self.waiting:
            waiting_reqs.append(await self.queue.get())

        while True:
            try:
                waiting_reqs.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        self.waiting.extend(waiting_reqs)

        selected_req = []
        fail_req = []
        while self.waiting and self.free_slots:
            req = self.waiting.pop(0)
            if req.computed_tokens:
                if self.kvcache.allocate(
                    req, _ceil_div(req.computed_tokens + 1, self.kv_block_size)
                ):
                    if decode_reqs is None:
                        decode_reqs = ScheduleReqs(requests=[], is_prefill=False)
                    slot_id = self.free_slots.pop()
                    self.req_to_slot[req.req_id] = slot_id
                    req.state = RequestState.RUNNING
                    decode_reqs.requests.append(req)
                else:
                    fail_req.append(req)
            else:
                if self.kvcache.allocate(
                    req, _ceil_div(len(req.input_tokens), self.kv_block_size)
                ):
                    slot_id = self.free_slots.pop()
                    self.req_to_slot[req.req_id] = slot_id
                    req.state = RequestState.RUNNING
                    selected_req.append(req)
                else:
                    fail_req.append(req)

        self.waiting = fail_req + self.waiting
        if selected_req:
            self.running.extend(selected_req)
            prefill_reqs = ScheduleReqs(requests=selected_req, is_prefill=True)

        if prefill_reqs is None and decode_reqs is None and self.waiting:
            raise RuntimeError(
                "KV cache block pool exhausted; no request can make progress"
            )

        return prefill_reqs, decode_reqs

    def _release_slot(self, req: Request):
        slot_id = self.req_to_slot.pop(req.req_id, None)
        if slot_id is None:
            return
        self.decode_block_table[slot_id].fill_(0)
        self.free_slots.append(slot_id)

    def _clear_decode_buffers(self):
        self.decode_input.fill_(self.dummy_token_id)
        self.decode_position_index.zero_()
        self.decode_active_mask.zero_()
        self.decode_block_table.zero_()

    def build_model_input(self, schedule_reqs: ScheduleReqs, device) -> ModelInput:
        flat_tokens = [
            tok for req in schedule_reqs.requests for tok in req.input_tokens
        ]
        flat_input = torch.tensor(flat_tokens, dtype=torch.long, device=device)

        input_lens = torch.tensor(
            [len(req.input_tokens) for req in schedule_reqs.requests],
            dtype=torch.long,
        )
        req_indptr_cpu = torch.cat(
            [torch.tensor([0], dtype=torch.long), torch.cumsum(input_lens, 0)]
        )

        total_tokens_num = req_indptr_cpu[-1]
        req_len_cpu = req_indptr_cpu[1:] - req_indptr_cpu[:-1]
        # Position inside each request. Shape: (total_tokens_num,)
        position_index = torch.arange(total_tokens_num) - torch.repeat_interleave(
            req_indptr_cpu[:-1],
            req_len_cpu,
        )
        position_index = position_index.to(device)
        slot_ids_cpu = [self.req_to_slot[req.req_id] for req in schedule_reqs.requests]

        for req in schedule_reqs.requests:
            slot_id = self.req_to_slot[req.req_id]
            self.decode_block_table[slot_id].fill_(0)
            blocks = self.kvcache.get_blocks(req)
            self.decode_block_table[slot_id, : len(blocks)] = torch.tensor(
                blocks,
                dtype=self.decode_block_table.dtype,
                device=self.decode_block_table.device,
            )

        return ModelInput(
            flat_input,
            req_indptr_cpu,
            position_index,
            slot_ids_cpu,
            None,
            self.decode_block_table,
        )

    def build_model_input_decode(self, schedule_reqs: ScheduleReqs) -> ModelInput:
        self._clear_decode_buffers()

        for req in schedule_reqs.requests:
            slot_id = self.req_to_slot[req.req_id]
            if self.max_seq_len is not None and req.computed_tokens >= self.max_seq_len:
                raise ValueError(
                    f"request {req.req_id} decode position {req.computed_tokens} "
                    f"must be less than max_seq_len {self.max_seq_len}"
                )
            self.decode_input[slot_id] = req.input_tokens[0]
            self.decode_position_index[slot_id] = req.computed_tokens
            self.decode_active_mask[slot_id] = True
            self.decode_block_table[slot_id].fill_(0)
            blocks = self.kvcache.get_blocks(req)
            self.decode_block_table[slot_id, : len(blocks)] = torch.tensor(
                blocks,
                dtype=self.decode_block_table.dtype,
                device=self.decode_block_table.device,
            )

        return ModelInput(
            self.decode_input,
            self.decode_req_indptr_cpu,
            self.decode_position_index,
            None,
            self.decode_active_mask,
            self.decode_block_table,
        )

    def _apply_token(self, req: Request, tok: int, stop_ids):
        req.computed_tokens += len(req.input_tokens)
        if tok in stop_ids:
            req.state = RequestState.FINISHED
            self.finished.append(req)
            return

        req.input_tokens = [tok]
        req.output_tokens.append(tok)
        if len(req.output_tokens) >= req.max_new_tokens:
            req.state = RequestState.FINISHED
            self.finished.append(req)

    def process_output_decode(self, schedule_reqs, gen_tokens: Tensor, stop_ids):
        gen_tokens_list = gen_tokens.detach().cpu().view(-1).tolist()
        for req in schedule_reqs.requests:
            slot_id = self.req_to_slot[req.req_id]
            tok = gen_tokens_list[slot_id]
            self._apply_token(req, tok, stop_ids)
        self.running = [
            req for req in self.running if req.state != RequestState.FINISHED
        ]

    def process_output(self, schedule_reqs, gen_tokens: Tensor, stop_ids):
        gen_tokens_list = gen_tokens.detach().cpu().view(-1).tolist()
        for req, tok in zip(schedule_reqs.requests, gen_tokens_list):
            self._apply_token(req, tok, stop_ids)
        self.running = [
            req for req in self.running if req.state != RequestState.FINISHED
        ]

    def drain_finished_reqs(self):
        finished = self.finished
        self.finished = []
        for req in finished:
            self._release_slot(req)
            self.kvcache.release(req)
        return finished

    def fail_all(self, error: Exception) -> list[Request]:
        done_reqs = []
        seen_req_ids = set()

        def add_finished(req: Request):
            if req.req_id in seen_req_ids:
                return
            seen_req_ids.add(req.req_id)
            done_reqs.append(req)

        def add_failed(req: Request):
            if req.req_id in seen_req_ids:
                return
            seen_req_ids.add(req.req_id)
            req.error = error
            req.state = RequestState.FINISHED
            done_reqs.append(req)

        for req in self.finished:
            add_finished(req)
        for req in self.running:
            add_failed(req)
        for req in self.waiting:
            add_failed(req)

        while True:
            try:
                add_failed(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        self.waiting = []
        self.running = []
        self.finished = []
        self.req_to_slot = {}
        self.free_slots = list(range(self.max_decode_slots))
        self._clear_decode_buffers()
        self.kvcache.reset()
        return done_reqs
