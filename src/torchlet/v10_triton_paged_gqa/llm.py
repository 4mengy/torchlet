import asyncio
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
from .request import Request
from .engine import Engine


class LLM:
    def __init__(
        self,
        model_id: str,
        max_decode_slots: int = 4,
        kv_block_size: int = 16,
        max_num_blocks: int | None = None,
    ):
        logger.info("pytorch info: \n%s", get_backend_info())

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        self.device = torch.device("cuda")

        model_dir = Path(snapshot_download(model_id))
        weights = self._load_weights(model_dir)
        logger.info("weight info: \n%s", get_weights_info(weights))

        self.config = json.loads((model_dir / "config.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.stop_ids = {
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        }
        self.stop_ids = {tok_id for tok_id in self.stop_ids if tok_id is not None}
        dummy_token_id = self.tokenizer.pad_token_id
        if dummy_token_id is None:
            dummy_token_id = self.tokenizer.eos_token_id
        if dummy_token_id is None:
            dummy_token_id = 0

        self.model = Qwen2ForCausalLM(self.config)
        load_result = self.model.load_state_dict(weights, strict=False)
        logger.info("missing_keys: %s", load_result.missing_keys)
        logger.info("unexpected_keys: %s", load_result.unexpected_keys)
        del weights
        self.model.to(self.device)
        self.model.eval()

        self.engine = Engine(
            self.model,
            self.tokenizer,
            self.device,
            self.stop_ids,
            max_decode_slots=max_decode_slots,
            dummy_token_id=dummy_token_id,
            kv_block_size=kv_block_size,
            max_num_blocks=max_num_blocks,
        )

    def _load_weights(self, model_dir: Path) -> dict[str, Tensor]:
        safetensors = sorted(model_dir.glob("*.safetensors"))

        if not safetensors:
            raise FileNotFoundError(f"not found safetensors files in {model_dir}")

        weights = {}
        for f in safetensors:
            weights.update(load_file(str(f)))
        return weights

    async def generate(self, inputs: list[str], max_new_tokens: int = 128) -> list[str]:
        if not inputs:
            return []
        if max_new_tokens <= 0:
            return [""] * len(inputs)

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
        req_batch = [
            Request(txt, tok, max_new_tokens=max_new_tokens)
            for txt, tok in zip(text, encoded["input_ids"])
        ]

        self.engine.start()
        await self.engine.add_requests(req_batch=req_batch)
        await asyncio.gather(*(req.done.wait() for req in req_batch))

        for req in req_batch:
            if req.error is not None:
                raise req.error

        return [req.output for req in req_batch]


if __name__ == "__main__":

    async def main():
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        llm = LLM(model_id)
        try:
            task1 = asyncio.create_task(
                llm.generate(["hello, do a simple introduction"], max_new_tokens=64)
            )
            await asyncio.sleep(0.1)

            task2 = asyncio.create_task(
                llm.generate(["what's the nearest star"], max_new_tokens=64)
            )
            await asyncio.sleep(0.1)

            task3 = asyncio.create_task(
                llm.generate(["write a short haiku about GPUs"], max_new_tokens=64)
            )

            outputs = await asyncio.gather(task1, task2, task3)
            for i, batch_outputs in enumerate(outputs, start=1):
                print(f"batch {i}:")
                print(batch_outputs)
        finally:
            await llm.engine.shutdown()

    asyncio.run(main())
