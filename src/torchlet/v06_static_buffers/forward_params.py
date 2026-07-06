from torch import Tensor
from .kvcache import KVCache
from dataclasses import dataclass


@dataclass
class ForwardParams:
    req_ids: list[str]
    req_indptr_cpu: Tensor
    position_index: Tensor
    active_mask_cpu: None | list[int]
    is_prefill: bool
    kvcache: KVCache
