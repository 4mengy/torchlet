import torch
from torch import Tensor


class KVCache:
    def __init__(
        self,
        max_layer,
        max_decode_slots,
        num_kv_heads,
        max_seq_len,
        head_dim,
        device,
        dtype,
    ):
        self.cache = torch.zeros(
            2,
            max_layer,
            max_decode_slots,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.max_seq_len = max_seq_len

        self.cache_positions = torch.arange(max_seq_len, device=device)

    def append_prefill(
        self,
        keys: Tensor,
        values: Tensor,
        layer_idx: int,
        slot_ids_cpu: list[int],
        req_indptr_cpu: Tensor,
    ):
        req_indptr_list = req_indptr_cpu.tolist()
        for i, slot_id in enumerate(slot_ids_cpu):
            start = req_indptr_list[i]
            end = req_indptr_list[i + 1]
            seq_len = end - start
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"prefill seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
                )
            self.cache[0, layer_idx, slot_id, :, 0:seq_len, :] = keys[:, start:end, :]
            self.cache[1, layer_idx, slot_id, :, 0:seq_len, :] = values[:, start:end, :]

    def append_decode(
        self,
        keys: Tensor,
        values: Tensor,
        layer_idx: int,
        position_index: Tensor,
        active_mask: Tensor,
    ):
        # (num_kv_heads, slot, head_dim) -> (slot, num_kv_heads, head_dim)
        keys_slot = keys.transpose(0, 1)
        vals_slot = values.transpose(0, 1)

        # (slots, num_kv_heads, max_seq_len, head_dim)
        k_layer = self.cache[0, layer_idx]
        v_layer = self.cache[1, layer_idx]
        max_decode_slots, num_kv_heads, _, head_dim = k_layer.shape

        index = position_index.view(max_decode_slots, 1, 1, 1).expand(
            max_decode_slots, num_kv_heads, 1, head_dim
        )

        old_k = k_layer.gather(dim=2, index=index)
        old_v = v_layer.gather(dim=2, index=index)

        mask = active_mask.view(max_decode_slots, 1, 1, 1)

        # (slot, num_kv_heads, 1, head_dim)
        k_src = torch.where(mask, keys_slot.unsqueeze(2), old_k)
        v_src = torch.where(mask, vals_slot.unsqueeze(2), old_v)

        k_layer.scatter_(dim=2, index=index, src=k_src)
        v_layer.scatter_(dim=2, index=index, src=v_src)

    def get_slot(self, layer_idx: int, slot_id):
        return self.cache[0, layer_idx, slot_id, :, :, :], self.cache[
            1, layer_idx, slot_id, :, :, :
        ]
