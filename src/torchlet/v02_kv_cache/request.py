import uuid
from dataclasses import dataclass, field
import torch
from torch import Tensor


@dataclass
class Request:
    prompt: str
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    done: bool = False
    req_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    computed_tokens: int = 0


class RequestBatch:
    def __init__(self, text: list[str], input_tokens: list[list[int]], stop_ids):
        self.requests = []
        for i in range(len(text)):
            self.requests.append(Request(text[i], input_tokens[i]))
        self.stop_ids = stop_ids
        self.active = list(self.requests)
        self.is_prefill = True

    def gen_llm_req(self, device):
        if not self.active:
            return (
                [],
                torch.empty(0, dtype=torch.long, device=device),
                torch.tensor([0], dtype=torch.long),
                torch.empty(0, dtype=torch.long, device=device),
            )

        flat_tokens = [tok for req in self.active for tok in req.input_tokens]
        flat_input = torch.tensor(flat_tokens, dtype=torch.long, device=device)

        input_lens = torch.tensor(
            [len(req.input_tokens) for req in self.active],
            dtype=torch.long,
        )
        req_indptr_cpu = torch.cat(
            [torch.tensor([0], dtype=torch.long), torch.cumsum(input_lens, 0)]
        )

        req_ids = [req.req_id for req in self.active]

        if self.is_prefill:
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
                [req.computed_tokens for req in self.active], device=device
            )
        return req_ids, flat_input, req_indptr_cpu, position_index

    def process_output(self, gen_tokens: Tensor):
        self.is_prefill = False
        gen_tokens_list = gen_tokens.detach().cpu().view(-1).tolist()
        if len(gen_tokens_list) != len(self.active):
            raise ValueError(
                f"gen_tokens length {len(gen_tokens_list)} does not match "
                f"active request count {len(self.active)}"
            )

        next_active = []
        for req, tok in zip(self.active, gen_tokens_list):
            req.computed_tokens += len(req.input_tokens)
            if tok in self.stop_ids:
                req.done = True
                continue

            req.input_tokens = [tok]
            req.output_tokens.append(tok)
            next_active.append(req)

        self.active = next_active
