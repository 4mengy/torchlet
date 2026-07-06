import asyncio
from .request import Request, RequestState
from dataclasses import dataclass
from torch import Tensor
import torch


@dataclass
class ScheduleReqs:
    requests: list[Request]
    is_prefill: bool


@dataclass
class ModelInput:
    req_ids: list[str]
    flat_input_ids: Tensor
    req_indptr_cpu: Tensor
    position_index: Tensor
    active_mask_cpu: None | list[int]


class Scheduler:
    def __init__(
        self,
        queue: asyncio.Queue[Request],
        device,
        max_decode_slots: int = 4,
        dummy_token_id: int = 0,
    ):
        if max_decode_slots <= 0:
            raise ValueError("max_decode_slots must be positive")
        self.waiting = list()
        self.running = list()
        self.finished = list()
        self.queue = queue
        self.max_decode_slots = max_decode_slots
        self.dummy_token_id = dummy_token_id
        self.slots = [None] * self.max_decode_slots
        self.req_to_slot = dict()
        self.free_slots = [i for i in range(0, self.max_decode_slots)]

        self.decode_input = torch.zeros(
            self.max_decode_slots, dtype=torch.long, device=device
        )
        self.decode_req_indptr_cpu = torch.arange(
            self.max_decode_slots + 1, dtype=torch.long
        )
        self.decode_position_index = torch.zeros(
            self.max_decode_slots, dtype=torch.long, device=device
        )
        self.decode_req_ids = [""] * self.max_decode_slots
        self.decode_active_mask_cpu = [0] * self.max_decode_slots

    async def add_requests(self, requests: list[Request]):
        for req in requests:
            await self.queue.put(req)

    async def schedule(self) -> tuple[ScheduleReqs | None, ScheduleReqs | None]:
        decode_reqs = None
        if self.running:
            decode_reqs = ScheduleReqs(requests=list(self.running), is_prefill=False)

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
        while self.waiting and self.free_slots:
            req = self.waiting.pop(0)
            slot_id = self.free_slots.pop()
            self.slots[slot_id] = req
            self.req_to_slot[req.req_id] = slot_id
            req.state = RequestState.RUNNING
            selected_req.append(req)

        if selected_req:
            self.running.extend(selected_req)
            prefill_reqs = ScheduleReqs(requests=selected_req, is_prefill=True)

        return prefill_reqs, decode_reqs

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

        req_ids = [req.req_id for req in schedule_reqs.requests]

        total_tokens_num = req_indptr_cpu[-1]
        req_len_cpu = req_indptr_cpu[1:] - req_indptr_cpu[:-1]
        # position_index shape: (total_tokens_num,)
        position_index = torch.arange(total_tokens_num) - torch.repeat_interleave(
            req_indptr_cpu[:-1],
            req_len_cpu,
        )
        position_index = position_index.to(device)
        return ModelInput(req_ids, flat_input, req_indptr_cpu, position_index, None)

    def build_model_input_decode(
        self, schedule_reqs: ScheduleReqs, device
    ) -> ModelInput:
        self.decode_input.fill_(self.dummy_token_id)
        self.decode_position_index.zero_()
        for slot_id in range(self.max_decode_slots):
            self.decode_req_ids[slot_id] = ""
            self.decode_active_mask_cpu[slot_id] = 0

        for req in schedule_reqs.requests:
            slot_id = self.req_to_slot[req.req_id]
            self.decode_input[slot_id] = req.input_tokens[0]
            self.decode_req_ids[slot_id] = req.req_id
            self.decode_position_index[slot_id] = req.computed_tokens
            self.decode_active_mask_cpu[slot_id] = 1

        return ModelInput(
            self.decode_req_ids,
            self.decode_input,
            self.decode_req_indptr_cpu,
            self.decode_position_index,
            self.decode_active_mask_cpu,
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
            slot_id = self.req_to_slot[req.req_id]
            self.free_slots.append(slot_id)
            self.req_to_slot.pop(req.req_id)
            self.slots[slot_id] = None
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
        self.slots = [None] * self.max_decode_slots
        self.req_to_slot = {}
        self.free_slots = list(range(self.max_decode_slots))
        return done_reqs
