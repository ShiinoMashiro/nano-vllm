# Week 1 拉通计划:nano-vllm × Qwen3-1.7B(详细执行版)

## 状态总览

| Phase | 内容 | 状态 | 产出 |
| --- | --- | --- | --- |
| 0 | 环境准备 | ✅ 2026-10-24 | torch 2.8.0+cu128 + flash-attn 2.8.3 + Qwen3-1.7B |
| 1 | 跑通生成 | ✅ 2026-10-24 | commit `0683669` |
| 2 | 基线数字 | ✅ 2026-10-24 | commit `a0661f9`,数字见 `docs/BASELINE.md` |
| 3 | greedy 支持 + tensor-diff 对数 | ⬜ | 预期:上游 PR 候选 + `tools/logdiff.py` |
| 4 | 开关/容量实验 | ✅ 2026-10-25 | 预期:BASELINE.md 补充表格 |

环境锚点(复现用):WSL2 | 3060 Ti 8GB | 驱动 591.86 | Python 3.12.3 | torch 2.8.0+cu128 | flash-attn 2.8.3.post2 | transformers 4.57.6 | venv:`~/workspace/venv`

关键源码事实(写计划时已核实):
- `LLM` = `LLMEngine`,构造参数白名单来自 `nanovllm/config.py::Config`:`max_num_batched_tokens / max_num_seqs / max_model_len / gpu_memory_utilization(默认 0.9) / tensor_parallel_size / enforce_eager / kvcache_block_size(256) / num_kvcache_blocks`
- **没有 prefix caching 开关**(block_manager 的 hash 前缀复用是常开逻辑)——量化它的收益只能靠 workload 对比
- `SamplingParams` 显式 `assert temperature > 1e-10, "greedy sampling is not permitted"`——greedy 是上游**未实现的功能**,不是 bug
- 模型是 thinking 模型:输出带 `<think>` 段,生成很长,对数实验必须考虑截断

---

## Phase 0:环境准备 ✅(已执行,存档)

实际执行顺序与命令(踩坑版,复现按此顺序):

```bash
# 1. torch 从 2.11+cu130 降到 2.8.0(原因:flash-attn 官方 wheel 只覆盖到 torch 2.8/cu12)
pip install torch==2.8.0
python -c "import torch; print(torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI)"   # → True

# 2. flash-attn 装"预编译 wheel"(绝不能让 pip 触发源码编译):abi=True → 选 cxx11abiTRUE
#    GitHub 直连 81KB/s,套镜像前缀后 ~100MB/s
wget -c "https://gh-proxy.com/https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" -O /tmp/flash_attn.whl
pip install /tmp/flash_attn.whl
python -c "from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache; print('OK')"

# 3. nano-vllm 本体(此时依赖全满足,不会回头碰 flash-attn)
pip install -e .

# 4. 模型(国内镜像)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3-1.7B --local-dir ~/huggingface/Qwen3-1.7B
```

教训(已记入 NOTES 素材):flash-attn 必须**先于** `pip install -e .` 安装;ABI 标志用 `torch._C._GLIBCXX_USE_CXX11_ABI` 确认;GitHub 下载套 `https://gh-proxy.com/` 前缀(时效性强,失效换 ghfast.top)。

遗留项:
- [ ] Windows 侧建 `C:\Users\yao\.wslconfig`(`[wsl2] memory=10GB / swap=8GB`)并 `wsl --shutdown` 生效——当前 VS Code Server 占 ~4.5GB,VM 内存紧张是之前 WSL 崩溃的诱因之一

## Phase 1:跑通生成 ✅(已执行,存档)

```bash
# example.py:path 指向 ~/huggingface/Qwen3-1.7B/(注意别指向空目录)
python example.py        # enforce_eager=False:graph capture 启动稍慢,decode 正常
# 再把 enforce_eager 改 True 跑一遍:启动快,decode 略慢
```

验证结果:两模式均正常输出;Qwen3-1.7B 为 thinking 模型,输出 `<think>` 推理段后再给答案。
单请求 decode ≈ 40~60 tok/s(eager,batch 2)。Commit:`0683669`。

## Phase 2:基线数字 ✅(已执行,存档)

详细步骤与数据见 `docs/BASELINE.md`,三行结论:
1. CUDA graph 收益随 batch 递减:batch 4 时 **+24.2%**,batch 32 时 ≈0;
2. 聚合吞吐 32→64 出现平台(1077→1080),推断 KV 容量上限(pool ≈4GB ÷ 112KB/token ≈ 36K token)→ 调度并发 ~35;
3. `dmon` 显示 `sm` 94~100%:GPU 忙满,host 不是当前瓶颈(与本职 LPU 框架结论形成对照)。

## Phase 3:greedy 采样支持 + tensor-diff 对数(2~3 天)

### 3.1 给 sampler 加 greedy(上游 feature PR 候选)

现状:`sampling_params.py` 用 assert 禁止 temperature≤0;`layers/sampler.py` 对所有请求走随机采样(gumbel 技巧),没有 argmax 路径。

- [ ] 从 `main` 拉分支:`git switch main && git switch -c feat/greedy-sampling`
- [ ] **先写失败复现**:`llm.generate(..., SamplingParams(temperature=0))` → 触发 AssertionError,贴进 NOTES
- [ ] 修改两个文件:
  - `sampling_params.py`:去掉 assert,改为 `assert self.temperature >= 0`(语义:0 = greedy)
  - `layers/sampler.py`:greedy 不能用 python 分支(会打断 @torch.compile),用张量掩码实现:
    ```python
    greedy = temperatures <= 0
    temps = torch.where(greedy, torch.ones_like(temperatures), temperatures)  # 防除零
    logits = logits.float().div_(temps.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    greedy_tokens = logits.argmax(dim=-1)          # 除以 1.0 不改变 argmax
    sample_tokens = torch.where(greedy, greedy_tokens, sampled)   # 支持混合 batch
    ```
- [ ] 验证:temperature=0 两次生成输出完全一致(确定性);temperature=1.0 行为与改前一致(回归)
- [ ] Commit:`feat: sampler 支持 greedy 采样(temperature=0)`
- [ ] **PR 准备**:push 分支到 fork → 向 `GeeeekExplorer/nano-vllm` 提 PR;正文写:动机(思考模型需要确定性输出/对数验证)、现状(assert 禁止)、实现(掩码式,兼容混合 batch、不破坏 torch.compile)、验证方式。这是你的第一个 upstream PR

### 3.2 logdiff 对数(tensor-diff v0.1)

- [ ] 新建 `tools/logdiff.py`,流程:
  1. 10 条 prompts:4 中文日常 + 3 英文 + 3 简单代码,**两侧用完全相同的 chat template**(`tokenizer.apply_chat_template`)
  2. golden 侧:HF `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16).cuda()`,`model.generate(do_sample=False, max_new_tokens=256)`(HF 原生支持 greedy)
  3. nano-vllm 侧:greedy(3.1 修好后 `temperature=0`),`max_tokens=256`
  4. 逐条比较 token id 序列:记录**首异位置**、共同前缀长度、两侧各自的完整输出
- [ ] 判定标准(写死在脚本里):greedy 下不要求全程一致(BF16 数值差异会在 top-2 接近处放大),报告**首异位置分布**;若 10 条的最短共同前缀 ≥ 128 token,视为对数通过
- [ ] 已知干扰源(写进脚本注释):①thinking 模型 `<think>` 内部多步数值累积,分歧大概率出现在 think 段;②两侧 tokenizer 配置必须同源;③nano-vllm 的 CUDA graph 路径与 eager 路径数值可能不同——对数固定用同一路径
- [ ] 输出 `docs/LOGDIFF_REPORT.md`(逐条表格:prompt / 首异位置 / 共同前缀 / 结论)
- [ ] Commit:`feat: tensor-diff 雏形,tools/logdiff.py 对数 HF golden`

## Phase 4:开关与容量实验(1~2 天)

> 核心目标:用实验回答 Phase 2 留下的两个问题,并量化 prefix cache 收益。
> 注意:Config 里没有 prefix cache 开关(常开),所以它的收益用 **workload 对比**测。

### 4.1 验证"KV 容量平台"假设(Phase 2 结论 2)

- [ ] 命令:`python bench.py 64` 基线(已知 ≈1080,平台)→ 改 `gpu_memory_utilization=0.95`(bench.py 的 LLM 构造加参数)重跑
- [ ] 判定:若 64 seqs 吞吐明显高于 1080 → 平台确由 KV 容量决定;若不变 → 平台另有原因(记录并分析)
- [ ] 顺带记录 `memory_reserved` 变化(pool 变大多少)

### 4.2 prefix cache 收益(共享前缀 workload 对比)

- [ ] 构造两组 workload(各 32 条):
  - A 组:32 条**互不相同**的长 prompt(各 ~768 token,随机文本)
  - B 组:**共享同一段 768 token 的"系统提示词"** + 各自不同的短问题(~64 token)
- [ ] 测量:Prefill 阶段的 tok/s(tqdm 进度条自带 Prefill 速率)或总 prefill 时间
- [ ] 预期:B 组从第 2 条起 prefill 命中缓存,prefill 速率显著高于 A 组;命中率与 block 对齐有关(768 = 3×256 block 整对齐)
- [ ] 记录进 BASELINE.md,prompt 构造脚本放 `tools/gen_prefix_bench.py`

### 4.3 补齐矩阵(选做)

- [ ] graph × batch(4/32)已有 Phase 2 数据,补 eager 的 batch 扫描(8/16/64)
- [ ] `max_num_seqs` 对平台的影响(64 seqs + max_num_seqs=32/64 对比)

### 4.4 提交

```bash
git add bench.py tools/ docs/BASELINE.md
git commit -m "bench: 验证 KV 容量平台假设,量化 prefix cache 收益"
git push
```

## DoD(本分支收口标准)

- [x] 生成跑通,eager/CUDA graph 双路径验证(Phase 1)
- [x] `docs/BASELINE.md` 可信基线数字(Phase 2)
- [x] greedy 采样可用,PR 分支已推(feat/greedy-sampling),待网页提 PR(Phase 3.1)
- [x] `tools/logdiff.py` 完成,跨栈对数标准修订并记录(Phase 3.2)
- [x] KV 平台假设证伪(计算饱和),prefix cache 收益 16.2x(Phase 4)
- [ ] `docs/NOTES.md` 记录全部踩坑(flash-attn/wsl 崩溃/镜像加速/temperature 限制)

## 约定

- Commit message:中文简述式,一相一事
- 模型权重、数据集文件不进 git;`egg-info/`、`*.log` 已在 .gitignore
- 公司代码/内部信息不进本仓库
- profiled(抓 trace)数字一律不进 BASELINE.md
