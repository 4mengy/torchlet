from dataclasses import dataclass, field
import torch
from torch import Tensor


@dataclass
class Request:
    prompt: str
    input_tokens: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)
    done: bool = False


class RequestBatch:
    def __init__(self, text: list[str], input_tokens: list[list[int]], stop_ids):
        self.requests = []
        for i in range(len(text)):
            self.requests.append(Request(text[i], input_tokens[i]))
        self.stop_ids = stop_ids
        self.active = list(self.requests)

    def gen_llm_req(self, device):
        if not self.active:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.tensor([0], dtype=torch.long),
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
        return flat_input, req_indptr_cpu

    def process_output(self, gen_tokens: Tensor):
        gen_tokens_list = gen_tokens.detach().cpu().view(-1).tolist()
        if len(gen_tokens_list) != len(self.active):
            raise ValueError(
                f"gen_tokens length {len(gen_tokens_list)} does not match "
                f"active request count {len(self.active)}"
            )

        next_active = []
        for req, tok in zip(self.active, gen_tokens_list):
            if tok in self.stop_ids:
                req.done = True
                continue

            req.input_tokens.append(tok)
            req.output_tokens.append(tok)
            next_active.append(req)

        self.active = next_active
