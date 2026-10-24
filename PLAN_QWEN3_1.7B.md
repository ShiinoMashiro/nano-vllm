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

## Phase 2:基线数字(1~2 天)

- [ ] 修改 `bench.py`:path 换 1.7B;**按 8GB 显存缩参**——默认 256 seqs × 1024 in/out 的 KV cache 会爆显存(估算:~114KB/token × 256 × 2048 ≈ 60GB),先降到 `num_seqs=32, len 100~512`,能跑稳再逐步加
- [ ] 记录:tokens/s、总耗时、`enforce_eager` on/off 对比
- [ ] 写入 `docs/BASELINE.md`,固定格式:GPU / 驱动 / torch 版本 / 参数配置 / 数字 / 日期
- [ ] Commit:`bench: 8GB 显存适配参数与 Qwen3-1.7B 基线数字`

## Phase 3:对数验证 = tensor-diff 雏形(2 天)

- [ ] 新建 `tools/logdiff.py`:
  - 固定 seed,10 条 prompts(中文日常 + 英文 + 简单代码),greedy(temperature=0)
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
