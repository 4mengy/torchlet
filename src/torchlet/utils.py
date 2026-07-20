import torch
from torch import Tensor


def load_model_weights(
    model: torch.nn.Module,
    weights: dict[str, Tensor],
    strict: bool = False,
):
    """Load checkpoint weights without silently casting them to FP32.

    PyTorch copies state-dict tensors into the module's existing parameter
    dtype. Qwen modules are constructed in FP32 by default, so a normal
    ``load_state_dict`` would upcast a BF16 checkpoint. Match each parameter to
    its checkpoint dtype first, while leaving FP32 buffers such as RoPE
    frequencies unchanged.
    """
    parameter_dtypes: dict[int, torch.dtype] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        source = weights.get(name)
        if source is None or not source.is_floating_point():
            continue
        parameter_id = id(parameter)
        previous_dtype = parameter_dtypes.get(parameter_id)
        if previous_dtype is not None and previous_dtype != source.dtype:
            raise ValueError(
                f"tied parameter {name} has conflicting checkpoint dtypes: "
                f"{previous_dtype} and {source.dtype}"
            )
        parameter_dtypes[parameter_id] = source.dtype

    with torch.no_grad():
        for parameter in model.parameters():
            checkpoint_dtype = parameter_dtypes.get(id(parameter))
            if checkpoint_dtype is not None and parameter.dtype != checkpoint_dtype:
                parameter.data = parameter.data.to(dtype=checkpoint_dtype)

    return model.load_state_dict(weights, strict=strict)


def get_weights_info(weights: dict[str, Tensor]) -> str:
    rows = [(key, str(value.shape), str(value.dtype)) for key, value in weights.items()]

    key_width = max(len(key) for key, _, _ in rows)
    shape_width = max(len(shape) for _, shape, _ in rows)

    return "\n".join(
        f"{key:<{key_width}}\t{shape:<{shape_width}}\t{dtype}"
        for key, shape, dtype in rows
    )


def get_backend_info() -> str:
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_available:
        device = "cuda"
    elif mps_available:
        device = "mps"
    else:
        device = "cpu"

    return "\n".join(
        [
            f"torch: {torch.__version__}",
            f"device: {device}",
            f"cuda: {cuda_available}",
            f"mps: {mps_available}",
            f"default_dtype: {torch.get_default_dtype()}",
            f"num_threads: {torch.get_num_threads()}",
        ]
    )
