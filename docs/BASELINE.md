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
