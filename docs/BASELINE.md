# Baseline

硬件:RTX 3060 Ti 8GB | 驱动 591.86 | torch 2.8.0+cu128 | WSL2 | 模型:Qwen3-1.7B BF16(无量化)
bench 参数:len 100~512 in / 100~512 out,ignore_eos,max_model_len=2048,seed(0)

## 数据

| 日期 | 模式 | num_seqs | 吞吐 tok/s(聚合) | 单请求视角 tok/s | 显存 reserved |
| --- | --- | --- | --- | --- | --- |
| 2026-10-24 | graph | 4 | 255.07 | 63.8 | 5.79 GB |
| 2026-10-24 | **eager** | **4** | **205.41** | 51.4 | 5.79 GB |
| 2026-10-24 | graph | 8 | 337.06 | 42.1 | 5.92 GB |
| 2026-10-24 | graph | 16 | 722.24 | 45.1 | 6.02 GB |
| 2026-10-24 | graph | 32 | 1077.35 | 33.7 | 6.48 GB |
| 2026-10-24 | eager | 32 | 1100.13 | 34.4 | 6.48 GB |
| 2026-10-24 | graph | 64 | 1080.22 | 16.9 | ≈同 32(未见 OOM) |

执行命令:`python bench.py <num_seqs> [eager]`(bench.py 支持 argv 传参)

## 结论

### 1. CUDA graph 收益随 batch 增大递减(已实验证实)
- batch 4:graph 255 vs eager 205 → **graph +24.2%**
- batch 32:graph 1077 vs eager 1100 → **≈0%(差异在噪声内)**
- 解释:graph 消除的是每步 kernel launch 的 host 开销;batch 大时每步 GPU 执行时间长,
  launch 开销被摊薄,收益趋零。**graph 的目标场景是小 batch 低延迟,不是大 batch 吞吐。**

### 2. 聚合吞吐随 batch 亚线性增长,32→64 出现平台
- 4→8:×1.32;8→16:×2.14;16→32:×1.49;**32→64:×1.00(平台)**
- 平台解释(待 trace 验证):KV pool 按剩余显存分配 ≈ 4GB,每 token 112KB →
  容量约 36K token;64 seqs × ~1000 token/seq = 65K 需求 → 调度器并发上限 ~35,
  实际并发与 32 相当 → 吞吐不变。64 能跑不 OOM 的原因正是 KV pool 自适应分配 + 调度排队。

### 3. GPU 侧忙满,host 不是当前瓶颈
- dmon(graph,32):`sm` 94~100%,`mem`(带宽占用)29→64%,功率 173~190W
- sm 打满说明计算/访存流水在跑,host 供给跟得上——与本职 LPU 框架"host 是瓶颈"的结论
  形成对照:不同框架、不同 batch 下瓶颈位置不同,必须实测。

### 4. 显存三视角(graph,32)
- allocated(张量真用)5.57 GB;reserved(池子)6.48 GB;峰值 6.12 GB
- reserved − allocated ≈ 0.9 GB 为 allocator 预留/碎片;nvidia-smi 看到的 fb ≈ reserved + 驱动底噪

## 待办(下一步用 trace 验证)

- [ ] 平台期假设:抓 64 seqs 场景的调度行为(实际并发数)
- [ ] batch 4 下 CPU-only trace:确认 host 开销构成(launch/采样回读/输入组装)

## Phase 4 实验(2026-10-25)

### 1. KV 容量平台假设:证伪

| gpu_memory_utilization | num_seqs=64 吞吐 | 显存 reserved |
| --- | --- | --- |
| 0.9 | 1137 tok/s | 6.75 GB |
| 0.95 | 1141 tok/s | 7.40 GB |

命令:`python bench.py 64 graph 0.9 / 0.95`。pool 扩大 0.65GB 吞吐不变 → **平台非 KV 容量**。
修正解释:计算饱和——`sm` 已 94~100%,batch 64 每步 ~56ms(双倍工作量÷双倍步时),
吞吐持平;带宽账(3.4GB/步 ÷ 448GB/s ≈ 130 步/s 上限)也远未触及(64 批仅 17.8 步/s)。
推论:1.7B 在 3060 Ti 上 batch≈32 即进入算力/占用饱和区,继续扩 KV 无意义,
提吞吐要降计算量(投机解码、kernel 融合、量化)。

### 2. prefix cache 收益:共享前缀场景 16.2x

| workload | 提交 tokens | 实算(命中后) | wall | 速率 |
| --- | --- | --- | --- | --- |
| a 无共享(32×768) | 24576 | 24576 | 4.62s | 5322 tok/s |
| b 共享前缀(1 填充+31 命中) | 25792 | ≈1984 | **0.30s** | 表观 85279 tok/s |

命令:`python tools/prefix_bench.py a / b`。实算减少 87%,墙钟加速 16.2x,
实测 0.30s 与"前缀全命中"理论值(~0.37s)吻合;跨请求命中有效(多轮对话/共享系统提示词场景)。

## int8 KVCache 拆小版实验(2026-10-25,feat/int8-kvcache 分支)

实现:存储 int8(逐 token 逐 head 动态 absmax/127 量化,Triton 写入 kernel)+ 读时 gather 反量化;
读路径统一走 flash_attn_varlen(gather 索引全层共享、纯 GPU 构建无同步);int8 模式限定 eager
(gather 形状依赖数据,与 CUDA graph 静态形状冲突——已在 config 断言)。

### 容量:×2 ✅
num_kvcache_blocks:bf16 **71** → int8 **142**(同显存同参数,`tools/kv_int8_check.py` 打印)。

### 精度:量化扰动不改语义
同栈 greedy 10 条全部在位置 2 分叉(近并列 logits 被量化误差翻转 → 轨迹合理分岔)。
语义抽检:两侧输出均连贯,int8 版甚至跳过 think 直达答案(轨迹差异 ≠ 质量退化)。
待做:wikitext PPL 严格评测。

### 性能:materialize 反量化方案 TPOT 回退(诚实记录)

| bs | bf16 eager TPOT ms | int8 eager TPOT ms | 变化 |
| --- | --- | --- | --- |
| 1 | 32.88 | 38.88 | +18% |
| 2 | 29.96 | 43.60 | +46% |
| 4 | 29.88 | 52.34 | +75% |
| 8 | 28.01 | 42.88 | +53% |
| 32 | 29.58 | 44.66 | +51% |

原因:省一半 KV 读带宽 < gather+cast+mul 反量化开销(每层 ~10 kernel × 28 层)。
结论:v1 定位 = **容量 ×2 + 精度验证工具**;吞吐收益必须做融合 int8 attention kernel
(dequant 在 kernel 内,即课程项目 4 的完整版),已列入后续 kernel 项。
