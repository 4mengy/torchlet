import torch
import triton
from torch import Tensor
from .kernel.kvcache import append_decode_kernel


class KVCache:
    def __init__(
        self,
        max_layer,
        max_num_blocks,
        block_size,
        num_kv_heads,
        max_seq_len,
        head_dim,
        device,
        dtype,
    ):
        self.cache = torch.zeros(
            2,
            max_layer,
            max_num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.max_num_blocks = max_num_blocks
        self.max_seq_len = max_seq_len
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.cache_positions = torch.arange(max_seq_len, device=device)

        self.req_to_blocks: dict[str, list[int]] = dict()
        # Block 0 is a safe sentinel for empty block-table entries.
        self.free_blocks = list(range(1, max_num_blocks))

    def allocate(self, req, block_num):
        cur = 0
        if req.req_id in self.req_to_blocks:
            cur = len(self.req_to_blocks[req.req_id])
        else:
            self.req_to_blocks.setdefault(req.req_id, [])
        new = block_num - cur
        if new <= 0:
            return True
        if len(self.free_blocks) < new:
            return False
        for _ in range(new):
            self.req_to_blocks[req.req_id].append(self.free_blocks.pop())
        return True

    def get_blocks(self, req):
        return self.req_to_blocks.get(req.req_id, [])

    def release(self, req):
        if req.req_id in self.req_to_blocks:
            blks = self.req_to_blocks.pop(req.req_id)
            self.free_blocks.extend(blks)

    def reset(self):
        self.req_to_blocks.clear()
        self.free_blocks = list(range(1, self.max_num_blocks))

    def append_prefill(
        self,
        keys: Tensor,
        values: Tensor,
        layer_idx: int,
        slot_ids_cpu: list[int],
        req_indptr_cpu: Tensor,
        block_table: Tensor,
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

            for logical_block_id, token_start in enumerate(
                range(start, end, self.block_size)
            ):
                block_id = block_table[slot_id, logical_block_id].item()
                token_end = min(token_start + self.block_size, end)
                self.cache[
                    0,
                    layer_idx,
                    block_id,
                    :,
                    : token_end - token_start,
                    :,
                ] = keys[:, token_start:token_end, :]
                self.cache[
                    1,
                    layer_idx,
                    block_id,
                    :,
                    : token_end - token_start,
                    :,
                ] = values[:, token_start:token_end, :]

    def append_decode(
        self,
        keys: Tensor,
        values: Tensor,
        layer_idx: int,
        position_index: Tensor,
        active_mask: Tensor,
        block_table: Tensor,
    ):
        # Physical block storage: (block_id, num_kv_heads, block_size, head_dim)
        k_layer = self.cache[0, layer_idx]
        v_layer = self.cache[1, layer_idx]

        assert keys.shape == values.shape
        assert keys.stride() == values.stride()
        assert keys.stride(-1) == 1
        assert block_table.stride(1) == 1

        grid = (position_index.shape[0], self.num_kv_heads)
        append_decode_kernel[grid](
            k_layer,
            v_layer,
            keys,
            values,
            position_index,
            active_mask,
            block_table,
            k_layer.stride(0),
            k_layer.stride(1),
            k_layer.stride(2),
            keys.stride(0),
            keys.stride(1),
            block_table.stride(0),
            BLOCK_SIZE=self.block_size,
            HEAD_DIM=self.head_dim,
            HEAD_DIM_PAD=triton.next_power_of_2(self.head_dim),
            num_warps=1,
        )

    def get_slot(self, layer_idx: int, slot, block_table):
        blocks = block_table[slot]
        # Gather physical blocks and flatten them back into logical token order.
        # [max_blocks_per_req, num_kv_heads, block_size, head_dim]
        k_blocks = self.cache[0, layer_idx, blocks, :, :, :]
        v_blocks = self.cache[1, layer_idx, blocks, :, :, :]
        k = (
            k_blocks.transpose(0, 1)
            .contiguous()
            .view(self.num_kv_heads, -1, self.head_dim)
        )
        v = (
            v_blocks.transpose(0, 1)
            .contiguous()
            .view(self.num_kv_heads, -1, self.head_dim)
        )
        return k[:, : self.max_seq_len, :], v[:, : self.max_seq_len, :]
