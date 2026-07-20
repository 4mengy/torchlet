from torch import Tensor, nn

from ..forward_params import ForwardParams
from ..layer.rms_norm import RmsNorm
from ..layer.transformer_block import TransformerBlock


def _config_get(config: dict, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _rope_theta(config: dict) -> float:
    rope_parameters = _config_get(config, "rope_parameters")
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        return rope_parameters["rope_theta"]
    return _config_get(config, "rope_theta", 1_000_000.0)


class Qwen2Model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.vocab_size = _config_get(config, "vocab_size")
        self.hidden_size = _config_get(config, "hidden_size")
        self.num_hidden_layers = _config_get(config, "num_hidden_layers")
        self.num_attention_heads = _config_get(config, "num_attention_heads")
        self.num_key_value_heads = _config_get(
            config, "num_key_value_heads", self.num_attention_heads
        )
        self.intermediate_size = _config_get(config, "intermediate_size")
        self.max_position_embeddings = _config_get(config, "max_position_embeddings")
        self.rms_norm_eps = _config_get(config, "rms_norm_eps", 1e-6)
        self.attention_bias = _config_get(config, "attention_bias", True)
        self.rope_theta = _rope_theta(config)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                max_position_embeddings=self.max_position_embeddings,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
                attention_bias=self.attention_bias,
                rms_norm_eps=self.rms_norm_eps,
                rope_theta=self.rope_theta,
                layer_idx=layer_idx,
            )
            for layer_idx in range(self.num_hidden_layers)
        )
        self.norm = RmsNorm(self.hidden_size, eps=self.rms_norm_eps)

    def forward(self, input_ids: Tensor, forward_params: ForwardParams) -> Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, forward_params)
        x = self.norm(x)
        return x


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.vocab_size = _config_get(config, "vocab_size")
        self.lm_head = nn.Linear(
            _config_get(config, "hidden_size"),
            self.vocab_size,
            bias=False,
        )

        if _config_get(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: Tensor, forward_params: ForwardParams, last_token_only=False
    ) -> Tensor:
        hidden_states = self.model(input_ids, forward_params)
        if not forward_params.is_prefill:
            return self.lm_head(hidden_states)
        if last_token_only:
            last_index = forward_params.req_indptr_cpu[1:] - 1
            return self.lm_head(hidden_states[last_index.to(hidden_states.device), :])
        return self.lm_head(hidden_states)
