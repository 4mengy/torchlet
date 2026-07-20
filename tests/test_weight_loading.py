import importlib

import pytest
import torch
from torch import nn

from torchlet.utils import load_model_weights


VERSIONS = [
    "v00_full_recompute",
    "v01_0_ragged_batch",
    "v01_1_split_gqa",
    "v02_kv_cache",
    "v03_request_states",
    "v04_continuous_batching",
    "v05_decode_slots",
    "v06_static_buffers",
    "v07_cuda_graph",
    "v08_paged_gqa_py",
    "v09_triton_basics",
    "v10_triton_paged_gqa",
]


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.projection = nn.Linear(4, 8, bias=False)
        self.projection.weight = self.embedding.weight
        self.norm = nn.LayerNorm(4)
        self.register_buffer(
            "frequencies", torch.ones(4, dtype=torch.float32), persistent=False
        )


def test_load_model_weights_preserves_checkpoint_dtype_and_tied_weights():
    model = TiedModel()
    weights = {
        "embedding.weight": torch.randn(8, 4, dtype=torch.bfloat16),
        "norm.weight": torch.randn(4, dtype=torch.bfloat16),
        "norm.bias": torch.randn(4, dtype=torch.bfloat16),
    }

    result = load_model_weights(model, weights)

    assert result.missing_keys == ["projection.weight"]
    assert model.embedding.weight.dtype == torch.bfloat16
    assert model.projection.weight is model.embedding.weight
    assert model.norm.weight.dtype == torch.bfloat16
    assert model.norm.bias.dtype == torch.bfloat16
    assert model.frequencies.dtype == torch.float32
    torch.testing.assert_close(model.embedding.weight, weights["embedding.weight"])


def test_load_model_weights_supports_mixed_parameter_dtypes():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    weights = {
        "0.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        "0.bias": torch.randn(4, dtype=torch.bfloat16),
        "1.weight": torch.randn(4, 4, dtype=torch.float16),
        "1.bias": torch.randn(4, dtype=torch.float16),
    }

    load_model_weights(model, weights, strict=True)

    assert model[0].weight.dtype == torch.bfloat16
    assert model[1].weight.dtype == torch.float16


@pytest.mark.parametrize("version", VERSIONS)
def test_qwen_version_loads_bf16_parameters_without_casting_rope_buffer(version):
    module = importlib.import_module(f"torchlet.{version}.model.qwen2_5")
    config = {
        "vocab_size": 16,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 8,
        "tie_word_embeddings": True,
    }
    source = module.Qwen2ForCausalLM(config)
    weights = {
        name: tensor.to(torch.bfloat16)
        for name, tensor in source.state_dict().items()
        if tensor.is_floating_point()
    }
    model = module.Qwen2ForCausalLM(config)

    load_model_weights(model, weights, strict=True)

    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert model.lm_head.weight is model.model.embed_tokens.weight
    rope = model.model.layers[0].self_attn.pos_emb
    assert rope.freqs.dtype == torch.float32
