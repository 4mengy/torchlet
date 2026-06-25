import json
from pathlib import Path

import torch
from torch import Tensor

from safetensors.torch import load_file
from modelscope import snapshot_download
from transformers import AutoTokenizer

from torchlet.logger import logger
from torchlet.utils import get_weights_info, get_backend_info

from .model.qwen2_5 import Qwen2ForCausalLM


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
        load_result = self.model.load_state_dict(weights, strict=False)
        del weights
        self.model.to(self.device)
        logger.info("missing_keys: %s", load_result.missing_keys)
        logger.info("unexpected_keys: %s", load_result.unexpected_keys)

    def _load_weights(self, model_dir: Path) -> dict[str, Tensor]:
        safetensors = sorted(model_dir.glob("*.safetensors"))

        if not safetensors:
            raise FileNotFoundError(f"not found safetensors files in {model_dir}")

        weights = {}
        for f in safetensors:
            weights.update(load_file(str(f)))
        return weights

    def generate(self, input: str, max_new_tokens: int = 128) -> str:
        self.model.eval()
        stop_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        stop_ids = {tok_id for tok_id in stop_ids if tok_id is not None}

        messages = [{"role": "user", "content": input}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
            self.device
        )  # [batch, num_tokens]

        out_tok_ids = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model.forward(
                    input_ids, last_token_only=True
                )  # [batch, vocab_size]
                last_tok_id = self.sample(logits)  # [1]
                if last_tok_id.item() in stop_ids:
                    break

                out_tok_ids.append(last_tok_id)
                input_ids = torch.cat(
                    [input_ids, last_tok_id.view(1, 1)],
                    dim=-1,
                )

        if not out_tok_ids:
            return ""

        out_tok_ids = torch.cat(out_tok_ids, dim=0)  # [new_tokens]
        # move to CPU and convert to python list for tokenizer.decode
        out_tok_ids = out_tok_ids.cpu().tolist()
        return self.tokenizer.decode(out_tok_ids, skip_special_tokens=True)

    def sample(self, logits: Tensor):
        return logits.argmax(dim=-1)


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    llm = LLM(model_id)
    print(llm.generate("hello, do a simple introduction"))
