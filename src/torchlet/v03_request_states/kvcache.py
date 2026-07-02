import torch
from torch import Tensor
from dataclasses import dataclass


@dataclass
class KVStore:
    # (num_kv_heads, seq_len, head_dim)
    k: Tensor
    v: Tensor


class KVCache:
    def __init__(self):
        # [reqid][layerid]
        self.cache = dict()

    def append(
        self,
        keys: Tensor,
        values: Tensor,
        layer_idx: int,
        req_ids: list[str],
        req_indptr_cpu: Tensor,
    ):
        req_indptr_list = req_indptr_cpu.tolist()
        for i, reqid in enumerate(req_ids):
            start = req_indptr_list[i]
            end = req_indptr_list[i + 1]
            req_cache = self.cache.setdefault(reqid, {})
            layer_cache = req_cache.get(layer_idx, None)
            if layer_cache is None:
                req_cache[layer_idx] = KVStore(
                    keys[:, start:end, :], values[:, start:end, :]
                )
            else:
                req_cache[layer_idx] = KVStore(
                    torch.cat((layer_cache.k, keys[:, start:end, :]), dim=1),
                    torch.cat((layer_cache.v, values[:, start:end, :]), dim=1),
                )

    def get(self, reqid: str, layer_idx: int):
        kvstore = self.cache[reqid][layer_idx]
        return kvstore.k, kvstore.v
