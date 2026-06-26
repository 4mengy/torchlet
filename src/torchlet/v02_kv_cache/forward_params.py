import torch
from torch import Tensor
from .kvcache import KVCache


class ForwardParams:
    def __init__(
        self,
        req_ids,
        req_indptr_cpu: Tensor,
        position_index: Tensor,
        is_prefill: bool,
        kvcache: KVCache,
    ):
        self.req_ids: list[str] = req_ids
        self.req_indptr_cpu = req_indptr_cpu
        self.position_index = position_index
        self.is_prefill: bool = is_prefill
        self.kvcache = kvcache
