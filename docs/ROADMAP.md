# nano-vllm 功能现状与优化路线

> 基于 2026-10-24 源码核实(非 README 转述)。所有"预期提升"为机制推算的假设区间,一律以实测为准;每完成一项,数字进 `docs/BASELINE.md`。

## 1. 功能现状(源码核实)

### 已支持

| 功能 | 位置/证据 | 说明 |
| --- | --- | --- |
| 离线批量生成 | `llm.py` → `engine/llm_engine.py` | `LLM.generate()`,无 serving/API 层 |
| Continuous batching | `engine/scheduler.py` | 请求级调度 |
| Chunked prefill | `scheduler.py`(上游 PR #218 刚并入) | 长 prompt 切块 |
| PagedAttention | `layers/attention.py` 用 `flash_attn_with_kvcache` + block table | block=256 |
| Prefix caching | `engine/block_manager.py::compute_hash`(hash 前缀复用) | **常开,无配置开关** |
| CUDA graph(decode) | `engine/model_runner.py::capture_cudagraph`,`graph_bs` 分档 replay | 仅 decode,prefill 在图外 |
| torch.compile | sampler 等模块装饰器 | host 侧算子融合 |
| Tensor parallel 2~8 | `llm_engine.py` spawn 多进程 + dist | |
| 显存自管理 | `Config.gpu_memory_utilization=0.9`,`num_kvcache_blocks` 自适应 | KV pool 按剩余显存分配 |

### 明确不支持(源码核实)

greedy/temperature=0(sampler 无 argmax 路径,`SamplingParams` assert 禁止)、top-p/top-k/重复惩罚、logprobs 返回、beam search、流式输出与 API server、量化 kernel、MoE 模型、投机解码、异步调度。

### 当前基线(详见 BASELINE.md)

graph:batch4=255 / batch16=722 / batch32=1077 tok/s;eager:batch4=205 / batch32=1100;
32→64 平台(≈1080);每步 D2H 同步存在(`model_runner.py:218` `.tolist()`)。

## 2. 优化路线(按优先级;每项:假设 → 改动 → 测评 → 预期)

### P0-1 Greedy 采样支持(Phase 3.1,功能+前置)

- **假设**:greedy 是对数验证(tensor-diff)的硬前置;argmax 比 gumbel 采样少两次全词表 op
- **改动**:`sampling_params.py` 去 assert + `sampler.py` torch.where 掩码实现(方案见 PLAN Phase 3.1)
- **测评**:①确定性(temperature=0 连跑两次 token 全同);②bench TPOT @ batch 1(预期 argmax 略快);③对数通过率(logdiff)
- **预期**:功能价值为主,性能影响 ≈1-2%

### P1-1 消除每步 D2H 同步 ★有实证,host 侧第一目标

- **证据**:`model_runner.py:218` `.tolist()` —— 每 decode 步把采样结果搬回 CPU 并同步,32 步 generation 里 host 被卡 32 次
- **改动**:pinned buffer + `non_blocking=True` 异步回读,下一步使用前再同步;或 argmax 留在 GPU、调度所需信息用设备端更新(工程量大,先做前者)
- **测评**:①CPU trace 前后对比 `Memcpy DtoH + Synchronize` 的每步耗时;②bench TPOT @ batch 1 和 32
- **预期**:batch 1 TPOT 改善 5~15%(小 batch 每步 GPU 时间短,同步占比高);batch 32 影响小(被 GPU 时间掩盖)——**收益曲线与 CUDA graph 相反方向,又是一对好故事**

### P1-2 Prefix cache 收益量化(Phase 4.2)

- **假设**:常开的 hash 前缀复用有效,但从未量化
- **改动**:无代码,做 workload 对比(32 条共享 768-token 前缀 vs 32 条互不相同)
- **测评**:Prefill tok/s(tqdm 自带)与总 prefill 时间;顺带验证 768=3×256 整 block 对齐
- **预期**:共享场景 prefill 时间降 60~90%(前缀全部命中跳过计算)

### P2-1 KV 容量平台突破(Phase 4.1 → KV INT8)

- **证据**:32→64 吞吐平台 ≈1080,推断 pool≈4GB ÷112KB ≈ 36K token 封顶并发 ~35
- **改动**:第一步只调 `gpu_memory_utilization=0.95`(零代码);第二步 KV cache INT8(quant/dequant 插在 attention 前后,写一个小 kernel 或用 torch 原生)
- **测评**:64 seqs 吞吐是否突破 1080;平台并发数;KV INT8 后加 wikitext PPL 评测精度损失
- **预期**:util 0.9→0.95 平台并发按比例上移;KV INT8 容量 ×2 → 并发 ×2,精度损失通常 <1%(需实测)

### P2-2 投机解码(Week 2~3 主菜,独立分支)

- **做法**:draft=Qwen3-0.6B,target=1.7B;draft 自回归 k 步 → target 一次 verify 全接受前缀;需要 greedy(P0-1)先落地
- **测评**:新增接受率统计埋点(每步接受 token 数/提议数)→ `docs/`:接受率、TPOT、吞吐 @ batch 1/4/32
- **预期**:同族 draft 的 greedy 接受率 60~80%;batch 1 TPOT 1.5~2×;**batch 增大收益递减**(verify 的 target forward 从 1 token 变 k+1 token,计算翻倍)——预期和 P1-1、graph 三条收益曲线拼出完整叙事
- **风险**:draft/target 分歧处理、KV/位置回滚逻辑、与 CUDA graph 的 shape 兼容

### P3 kernel 层(学习件,不承诺收益)

| 项 | 测评 | 预期 |
| --- | --- | --- |
| Triton RMSNorm/RoPE/SwiGLU 对比 torch.compile | 每步 kernel 数(trace)+ TPOT | ≈0%(compile 已融合),价值是能力证明 |
| Attention 后端对比(flash-attn vs SDPA vs FlashInfer decode) | TPOT @ 上下文 512/2K/8K(1.7B 单序列 8K 仅 ~0.9GB KV,可测) | 短上下文≈0;长上下文 attention 占比升,或有 5~20% |
| mini fused MoE(OLMoE 线) | 对数 + fused vs unfused 吞吐 | 功能性 |

### P4 远期(不排期)

MoE 模型支持(OLMoE-1B-7B 进 engine)、W4A16 量化 kernel(sm86 走 marlin 类路径)、API server + streaming、EAGLE 风格 draft 头。

## 3. 测评纪律(所有优化共用)

1. **单变量**:一次只改一个开关;graph/eager 分别记录
2. **三件套数字**:聚合吞吐、单请求视角(聚合÷batch)、显存 allocated/reserved
3. **固定负载**:seed(0)、长度分布不变;profiled 数字不进表
4. **先假设后实验**:每项先写"预期与机制",跑完写"实测与解释"——不一致也是结论
5. 精度类优化(KV INT8)必须附 PPL/对数证据,不许只报速度

## 4. 推荐执行顺序

P0-1(半天)→ P1-1(1 天,有实证、改动小、收益可测)→ P2-2 投机解码(1~2 周,主菜)→ P1-2/P2-1(穿插)→ P3 穿插学习件。
P1-1 和 P2-2 都依赖 greedy,这也是 Phase 3.1 排最前的原因。
