import json
from pathlib import Path

import torch
from torch import Tensor

from safetensors.torch import load_file
from modelscope import snapshot_download
from transformers import AutoTokenizer

from torchlet.logger import logger
from torchlet.utils import get_backend_info, get_weights_info, load_model_weights

from .forward_params import ForwardParams
from .model.qwen2_5 import Qwen2ForCausalLM
from .request import Request
from .kvcache import KVCache
from .scheduler import Scheduler


class LLM:
    def __init__(self, model_id: str):
        logger.info("pytorch info: \n%s", get_backend_info())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_dir = Path(snapshot_download(model_id))
        weights = self._load_weights(model_dir)
        logger.info("weight info: \n%s", get_weights_info(weights))

        self.config = json.loads((model_dir / "config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = Qwen2ForCausalLM(self.config)
        load_result = load_model_weights(self.model, weights)
        logger.info("missing_keys: %s", load_result.missing_keys)
        logger.info("unexpected_keys: %s", load_result.unexpected_keys)
        del weights
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, model_dir: Path) -> dict[str, Tensor]:
        safetensors = sorted(model_dir.glob("*.safetensors"))

        if not safetensors:
            raise FileNotFoundError(f"not found safetensors files in {model_dir}")

        weights = {}
        for f in safetensors:
            weights.update(load_file(str(f)))
        return weights

    def generate(self, inputs: list[str], max_new_tokens: int = 128) -> list[str]:
        stop_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        stop_ids = {tok_id for tok_id in stop_ids if tok_id is not None}

        messages = [[{"role": "user", "content": input}] for input in inputs]
        text = [
            self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in messages
        ]
        encoded = self.tokenizer(text, padding=False)

        req_batch = [Request(txt, tok) for txt, tok in zip(text, encoded["input_ids"])]
        scheduler = Scheduler()
        scheduler.add_request(req_batch)

        kvcache = KVCache()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                schedule_reqs = scheduler.schedule()
                if schedule_reqs is None:
                    break
                model_input = scheduler.build_model_input(schedule_reqs, self.device)

                if model_input.flat_input_ids.numel() == 0:
                    break

                forward_params = ForwardParams(
                    req_ids=model_input.req_ids,
                    req_indptr_cpu=model_input.req_indptr_cpu,
                    position_index=model_input.position_index,
                    is_prefill=schedule_reqs.is_prefill,
                    kvcache=kvcache,
                )

                logits = self.model.forward(
                    model_input.flat_input_ids,
                    forward_params,
                    last_token_only=True,
                )  # [batch, vocab_size]
                gen_tok_id = self.sample(logits)  # [batch]
                scheduler.process_output(schedule_reqs, gen_tok_id, stop_ids)

        out_tok_ids = [req.output_tokens for req in req_batch]
        return self.tokenizer.batch_decode(out_tok_ids, skip_special_tokens=True)

    def sample(self, logits: Tensor):
        return logits.argmax(dim=-1)


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    llm = LLM(model_id)
    print(llm.generate(["hello, do a simple introduction", "what's the nearest star"]))
