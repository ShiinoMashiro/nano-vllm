# Week 1 拉通计划:nano-vllm × Qwen3-1.7B

> 目标:在本机(RTX 3060 Ti 8GB)跑通 Qwen3-1.7B 生成 → 建立基线数字 → 完成与 HF transformers 的对数验证(tensor-diff 雏形)。
> 原则:每个阶段至少一个 commit;模型权重不进 git;所有数字记录进 `docs/`。

## Phase 0:环境准备(0.5 天)

- [ ] 验证 CUDA:`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
- [ ] 安装依赖:`pip install -e .`(基于 pyproject.toml)
- [ ] 下载模型(BF16,约 3.4GB;国内走镜像):
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download Qwen/Qwen3-1.7B --local-dir ~/huggingface/Qwen3-1.7B
  ```
- [ ] Commit:`docs: Week1 拉通计划`

## Phase 1:跑通生成(1 天)

- [ ] `example.py` 的 `path` 改为 `~/huggingface/Qwen3-1.7B/`
- [ ] 先跑 `enforce_eager=True` 拿到生成结果;再跑 `enforce_eager=False`(CUDA graph 路径)
- [ ] 遇到的问题与解决记入 `docs/NOTES.md`(环境/报错/修复方式)
- [ ] Commit:`feat: 拉通 Qwen3-1.7B 生成(eager + cuda graph)`

## Phase 2:基线数字(1~2 天)——详细执行版

> 纪律:基线必须是**不 profile 的干净数字**;跑重活前关掉 VS Code(它占 VM 约 4.5GB 内存)。

### 2.1 改 `bench.py`(4 处)

```python
def main():
    seed(0)
    num_seqs = 32            # ① 256 → 32
    max_input_len = 512      # ② 1024 → 512
    max_ouput_len = 512      # ③ 1024 → 512(原代码拼写就是 ouput,别"修正")

    path = os.path.expanduser("~/huggingface/Qwen3-1.7B/")   # ④
    llm = LLM(path, enforce_eager=False, max_model_len=2048) # 顺带 4096 → 2048
```

为什么必须缩:Qwen3-1.7B 每 token KV ≈ 2×28层×8kv头×128维×2字节 ≈ 112KB。
默认 256 seqs×~2048 token ≈ **58GB KV** → 必爆;32 seqs×~1024 ≈ **3.5GB** → 权重 3.4GB + KV 3.5GB ≈ 7GB,贴着 8GB 上限。

### 2.2 跑基线(CUDA graph 模式)

```bash
python bench.py
```

- 记录输出:`Total: X tok, Time: Y s, Throughput: Z tok/s`
- 另开终端观察:`nvidia-smi dmon -s pum`(重点 `fb` 显存、`sm` 利用率)
- **OOM 处理**:`num_seqs` 减到 16 重试;`max_model_len` 降到 1024

### 2.3 跑对照(eager 模式)

```python
llm = LLM(path, enforce_eager=True, max_model_len=2048)   # 唯一改动
```

```bash
python bench.py
```

两组数字的差 = **CUDA graph 的收益,第一次量化优化**。

### 2.4 batch 扫描(选做,10 分钟)

`num_seqs` 依次取 8 / 16 / 32 / 64(graph 模式),记录吞吐:

- 预期:batch 增大 → 聚合吞吐上升(权重读取被摊薄),单请求速度 = 聚合÷batch 下降
- 这是"decode 是 bandwidth-bound"的实验证据

### 2.5 显存三视角(加进 bench.py 末尾)

```python
import torch
print(f"已分配(张量真用): {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"已预留(池子):     {torch.cuda.memory_reserved()/1e9:.2f} GB")  # ≈ nvidia-smi 的 fb
print(f"峰值分配:         {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

nvidia-smi 显示的是 reserved,不等于真用——OOM 诊断要看 allocated。

### 2.6 写 `docs/BASELINE.md`

```markdown
# Baseline

硬件:RTX 3060 Ti 8GB | 驱动 591.86 | torch 2.8.0+cu128 | WSL2
模型:Qwen3-1.7B BF16 | 精度:无量化

| 日期 | 模式 | num_seqs | in/out len | 吞吐(tok/s) | 备注 |
| --- | --- | --- | --- | --- | --- |
| 2026-10-XX | graph | 32 | 100~512 / 100~512 | | 聚合吞吐 |
| 2026-10-XX | eager | 32 | 同上 | | graph 收益 = (A-B)/B |
| 2026-10-XX | graph | 8/16/64 | 同上 | | batch 扫描 |

定义:吞吐为 N 条并行的聚合值;单请求视角 = 聚合 ÷ num_seqs(≈1/TPOT)。
profiled(抓 trace 时)的数字一律不进本表。
```

### 2.7 提交

```bash
git add bench.py docs/BASELINE.md
git commit -m "bench: 8GB 显存适配参数,Qwen3-1.7B 基线数字(graph vs eager + batch 扫描)"
git push
```

### 2.8 自查(面试 checkpoint)

跑通后应能回答:①decode 为什么 bandwidth-bound(3.4GB 权重每步重读 ÷ 448GB/s ≈ 理论上限);
②batch 增大为什么聚合涨、单条降;③KV cache 112KB/token 怎么算出来的。

## Phase 3:对数验证 = tensor-diff 雏形(2 天)

> ⚠️ 前置:nano-vllm 的 sampler 不支持 temperature=0(直接除零,见 NOTES)。两个选择:
> A.【推荐】给 sampler 加 greedy 路径(temperature≤0 → argmax)——同时是给上游提 PR 的素材;
> B. 临时方案:temperature>0 + `torch.manual_seed(seed)` 固定随机性(同 batch 形状下可复现)。

- [ ] 新建 `tools/logdiff.py`:
  - 固定 seed,10 条 prompts(中文日常 + 英文 + 简单代码),greedy(方案 A)或固定种子采样(方案 B)
  - HF transformers 生成作为 golden
  - nano-vllm 生成对比:**首异 token 位置 + 逐条一致率**(对数标准:greedy 下 token 完全一致,或明确说明前 N token 一致)
  - 输出结构化报告(MD 表格)
- [ ] 不一致 case 分析记入 `docs/NOTES.md`(采样实现差异?kernel 数值差异?)
- [ ] Commit:`feat: tensor-diff 雏形,tools/logdiff.py 对数 HF golden`

## Phase 4:开关实验(1 天,选做)

- [ ] 吞吐对比表:CUDA graph on/off × prefix caching on/off(利用 LLM 构造参数)
- [ ] 结果补进 `docs/BASELINE.md`——这是"优化必有前后数字"纪律的第一次演练

## DoD(本分支收口标准)

- [ ] 生成跑通,eager/CUDA graph 两条路径都验证
- [ ] `docs/BASELINE.md` 有可信基线数字
- [ ] `tools/logdiff.py` 对数通过(或差异已被解释)
- [ ] `docs/NOTES.md` 记录全部踩坑

## 约定

- Commit message:中文简述式,一相一事
- 模型权重、数据集文件不进 git(`.gitignore` 确认覆盖)
- 公司代码/内部信息不进本仓库
