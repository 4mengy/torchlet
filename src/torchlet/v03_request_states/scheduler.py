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


class Scheduler:
    def __init__(self):
        self.waiting = list()
        self.running = list()
        self.finished = list()

    def add_request(self, requests: list[Request]):
        self.waiting.extend(requests)

    def schedule(self) -> ScheduleReqs | None:
        if self.waiting:
            for req in self.waiting:
                req.state = RequestState.RUNNING
            schedule_reqs = ScheduleReqs(requests=self.waiting, is_prefill=True)
            self.running.extend(self.waiting)
            self.waiting = []
            return schedule_reqs

        if not self.running:
            return None

        return ScheduleReqs(requests=self.running, is_prefill=False)

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

        if schedule_reqs.is_prefill:
            total_tokens_num = req_indptr_cpu[-1]
            req_len_cpu = req_indptr_cpu[1:] - req_indptr_cpu[:-1]
            # position_index shape: (total_tokens_num,)
            position_index = torch.arange(total_tokens_num) - torch.repeat_interleave(
                req_indptr_cpu[:-1],
                req_len_cpu,
            )
            position_index = position_index.to(device)
        else:
            # (req_num,)
            position_index = torch.tensor(
                [req.computed_tokens for req in schedule_reqs.requests], device=device
            )
        return ModelInput(req_ids, flat_input, req_indptr_cpu, position_index)

    def process_output(self, schedule_reqs, gen_tokens: Tensor, stop_ids):
        gen_tokens_list = gen_tokens.detach().cpu().view(-1).tolist()
        new_running = []
        for req, tok in zip(schedule_reqs.requests, gen_tokens_list):
            req.computed_tokens += len(req.input_tokens)
            if tok in stop_ids:
                req.state = RequestState.FINISHED
                self.finished.append(req)
                continue
            else:
                new_running.append(req)
            req.input_tokens = [tok]
            req.output_tokens.append(tok)
        self.running = new_running
