import torch
from torch import Tensor


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
