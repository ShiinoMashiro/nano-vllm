import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


@triton.jit
def _store_kvcache_int8_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,
    H: tl.constexpr, HD: tl.constexpr,
):
    # 逐 token 逐 head 动态对称量化(absmax/127),行粒度与 paged 写入天然对齐
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    h = tl.arange(0, H)[:, None]
    d = tl.arange(0, HD)[None, :]
    k = tl.load(key_ptr + idx * key_stride + h * HD + d).to(tl.float32)
    v = tl.load(value_ptr + idx * value_stride + h * HD + d).to(tl.float32)
    ks = tl.maximum(tl.max(tl.abs(k), axis=1) / 127.0, 1e-8)
    vs = tl.maximum(tl.max(tl.abs(v), axis=1) / 127.0, 1e-8)
    off = slot * (H * HD) + h * HD + d
    tl.store(k_cache_ptr + off, (k / ks[:, None]).to(tl.int8))
    tl.store(v_cache_ptr + off, (v / vs[:, None]).to(tl.int8))
    srow = slot * H + tl.arange(0, H)
    tl.store(k_scale_ptr + srow, ks)
    tl.store(v_scale_ptr + srow, vs)


def store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping):
    N, H, HD = key.shape
    _store_kvcache_int8_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                     k_cache, v_cache, k_scale, v_scale, slot_mapping,
                                     H=H, HD=HD)


def dequant_gather_rows(int8_cache, scale, rows_idx):
    # int8 行 gather + 逐 head scale 反量化:(R, H*HD) int8 × (R, H) scale -> (R, H, HD) bf16
    R, HD_total = int8_rows.shape
    H = scale.size(1)
    HD = HD_total // H
    k = int8_rows.view(R, H, HD).to(scale.dtype)
    return k * scale[rows_idx].unsqueeze(-1)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and k_cache.dtype == torch.int8:
            store_kvcache_int8(k, v, k_cache, v_cache, self.k_scale, self.v_scale, context.slot_mapping)
            if context.is_prefill and context.block_tables is None:    # 无前缀缓存,当前 K/V 均为寄存器中的 bf16
                o = flash_attn_varlen_func(q, k, v,
                                           max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                           max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                           softmax_scale=self.scale, causal=True)
            else:    # 从 int8 cache 反量化读(model_runner 已把全层共享的 gather 索引放进 context)
                H = self.num_kv_heads
                HD_total = H * self.head_dim
                idx = context.kv_rows_idx
                R = idx.size(0)
                k_rows = k_cache.view(-1, HD_total)[idx].view(R, H, self.head_dim).to(q.dtype) * \
                         self.k_scale.view(-1, H)[idx].unsqueeze(-1)
                v_rows = v_cache.view(-1, HD_total)[idx].view(R, H, self.head_dim).to(q.dtype) * \
                         self.v_scale.view(-1, H)[idx].unsqueeze(-1)
                cu_q = context.cu_seqlens_q if context.is_prefill else \
                    torch.arange(q.size(0) + 1, dtype=torch.int32, device=q.device)
                o = flash_attn_varlen_func(q, k_rows, v_rows,
                                           max_seqlen_q=context.max_seqlen_q if context.is_prefill else 1,
                                           cu_seqlens_q=cu_q,
                                           max_seqlen_k=context.kv_max_len, cu_seqlens_k=context.kv_cu_k,
                                           softmax_scale=self.scale, causal=True)
            return o
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
