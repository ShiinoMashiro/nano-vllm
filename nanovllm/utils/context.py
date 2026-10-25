from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    kv_rows_idx: torch.Tensor | None = None    # int8 KV:全层共享的行 gather 索引
    kv_cu_k: torch.Tensor | None = None        # int8 KV:varlen 的 cu_seqlens_k
    kv_max_len: int = 0                        # int8 KV:max_seqlen_k 上界(免 .item() 同步)

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, kv_rows_idx=None, kv_cu_k=None, kv_max_len=0):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, kv_rows_idx, kv_cu_k, kv_max_len)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
