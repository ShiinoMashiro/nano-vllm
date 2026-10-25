import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 持久化 pinned/GPU buffer:消除 decode 每步的 pinned 分配(cudaHostAlloc)与中间张量
        self.dec_max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        with torch.device("cpu"):
            pin = dict(
                input_ids=torch.empty(self.dec_max_bs, dtype=torch.int64, pin_memory=True),
                positions=torch.empty(self.dec_max_bs, dtype=torch.int64, pin_memory=True),
                slot_mapping=torch.empty(self.dec_max_bs, dtype=torch.int32, pin_memory=True),
                context_lens=torch.empty(self.dec_max_bs, dtype=torch.int32, pin_memory=True),
                block_tables=torch.empty(self.dec_max_bs, max_num_blocks, dtype=torch.int32, pin_memory=True),
                temperatures=torch.empty(self.dec_max_bs, dtype=torch.float32, pin_memory=True),
                tokens=torch.empty(self.dec_max_bs, dtype=torch.int64, pin_memory=True),
            )
        gpu = {k: torch.empty(v.shape, dtype=v.dtype, device="cuda") for k, v in pin.items() if k != "tokens"}
        self.dec_pin, self.dec_gpu = pin, gpu
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        itemsize = 1 if config.kv_cache_dtype == "int8" else hf_config.dtype.itemsize
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        if config.kv_cache_dtype == "int8":
            self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, dtype=torch.int8)
            self.kv_scale = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks * self.block_size, num_kv_heads, dtype=hf_config.dtype)
        else:
            self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if config.kv_cache_dtype == "int8":
                    module.k_scale = self.kv_scale[0, layer_id]
                    module.v_scale = self.kv_scale[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables


    def build_kv_gather(self, lens: torch.Tensor, block_tables: torch.Tensor):
        # 把 (B, maxb) block_table 展开成 seq-major 的行索引,并给出 varlen 的 cu_seqlens_k。
        # 上界行数 R = B*maxb*block_size(含 padding,attention 只读 [0, cu[-1]) 区间)。
        B, maxb = block_tables.shape
        bsz = self.block_size
        cu = torch.zeros(B + 1, dtype=torch.int32, device=lens.device)
        torch.cumsum(lens, 0, dtype=torch.int32, out=cu[1:])
        ar = torch.arange(B * maxb * bsz, device=lens.device, dtype=torch.int32)
        seq_id = torch.searchsorted(cu[1:], ar, right=True).clamp(max=B - 1)
        token_pos = ar - cu[seq_id]
        valid = token_pos < lens[seq_id]
        block_pos = (token_pos // bsz).clamp(max=maxb - 1)
        idx = block_tables[seq_id, block_pos] * bsz + token_pos % bsz
        idx = torch.where(valid, idx, 0)
        return idx, cu

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        kv_kwargs = {}
        if self.config.kv_cache_dtype == "int8" and block_tables is not None:
            lens = torch.tensor(cu_seqlens_k[1:], dtype=torch.int32, device="cuda") - \
                   torch.tensor(cu_seqlens_k[:-1], dtype=torch.int32, device="cuda")
            idx, cu = self.build_kv_gather(lens, block_tables)
            kv_kwargs = dict(kv_rows_idx=idx, kv_cu_k=cu, kv_max_len=self.config.max_model_len)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, **kv_kwargs)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        bs = len(seqs)
        if bs > self.dec_max_bs:    # 超大 batch 退回逐次分配路径
            return self._prepare_decode_slow(seqs)
        pin, gpu = self.dec_pin, self.dec_gpu
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        block_tables = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            block_tables.append(seq.block_table)
        max_len = max(len(b) for b in block_tables)
        pin["input_ids"][:bs] = torch.tensor(input_ids, dtype=torch.int64)
        pin["positions"][:bs] = torch.tensor(positions, dtype=torch.int64)
        pin["slot_mapping"][:bs] = torch.tensor(slot_mapping, dtype=torch.int32)
        pin["context_lens"][:bs] = torch.tensor(context_lens, dtype=torch.int32)
        pin["block_tables"][:bs, :max_len] = torch.tensor(
            [b + [-1] * (max_len - len(b)) for b in block_tables], dtype=torch.int32)
        for k in ("input_ids", "positions", "slot_mapping", "context_lens"):
            gpu[k][:bs].copy_(pin[k][:bs], non_blocking=True)
        gpu["block_tables"][:bs, :max_len].copy_(pin["block_tables"][:bs, :max_len], non_blocking=True)
        bt = gpu["block_tables"][:bs, :max_len]
        if self.config.kv_cache_dtype == "int8":
            idx, cu = self.build_kv_gather(gpu["context_lens"][:bs], bt)
            set_context(False, slot_mapping=gpu["slot_mapping"][:bs],
                       context_lens=gpu["context_lens"][:bs], block_tables=bt,
                       kv_rows_idx=idx, kv_cu_k=cu, kv_max_len=self.config.max_model_len)
        else:
            set_context(False, slot_mapping=gpu["slot_mapping"][:bs],
                       context_lens=gpu["context_lens"][:bs], block_tables=bt)
        return gpu["input_ids"][:bs], gpu["positions"][:bs]

    def _prepare_decode_slow(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        if self.config.kv_cache_dtype == "int8":
            idx, cu = self.build_kv_gather(context_lens, block_tables)
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables,
                       kv_rows_idx=idx, kv_cu_k=cu, kv_max_len=self.config.max_model_len)
        else:
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        if len(seqs) <= self.dec_max_bs:
            bs = len(seqs)
            self.dec_pin["temperatures"][:bs] = torch.tensor(temperatures, dtype=torch.float32)
            self.dec_gpu["temperatures"][:bs].copy_(self.dec_pin["temperatures"][:bs], non_blocking=True)
            return self.dec_gpu["temperatures"][:bs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.rank == 0:
            tokens_gpu = self.sampler(logits, temperatures)
            bs = logits.size(0)
            if bs <= self.dec_max_bs and not is_prefill:
                # 异步 D2H 到常驻 pinned buffer,消除每步 pageable 分配
                self.dec_pin["tokens"][:bs].copy_(tokens_gpu, non_blocking=True)
                torch.cuda.current_stream().synchronize()
                token_ids = self.dec_pin["tokens"][:bs].tolist()
            else:
                token_ids = tokens_gpu.tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
