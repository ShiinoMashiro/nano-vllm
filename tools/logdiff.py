"""tensor-diff v0.1:nano-vllm greedy 输出 vs HF transformers golden 的 token 级对数。

判定标准:greedy 下不要求全程一致(BF16 数值差异会在 top-2 接近处放大),
以"首异位置分布"报告;10 条中最短共同前缀 >= 128 视为对数通过。
运行:python tools/logdiff.py
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-1.7B")
MAX_TOKENS = 256
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")   # sdpa / flash_attention_2
REPORT = os.path.join(os.path.dirname(__file__), "..", "docs", f"LOGDIFF_REPORT_{ATTN_IMPL}.md")

PROMPTS = [
    "用一句话介绍你自己",
    "写一首关于秋天的五言绝句",
    "推荐三本适合初学者的编程书籍,并说明理由",
    "解释什么是大语言模型的KV缓存",
    "Explain the difference between TCP and UDP in one paragraph.",
    "What causes the seasons on Earth?",
    "Write a Python function that checks if a number is prime.",
    "用Python实现快速排序,附注释",
    "Write a SQL query to find the second highest salary from an employees table.",
    "用bash写一个统计当前目录下文件总数的命令并解释",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL)
chat = [tokenizer.apply_chat_template([{"role": "user", "content": p}],
                                      tokenize=False, add_generation_prompt=True)
        for p in PROMPTS]

# ---- golden:HF transformers greedy ----
print("== HF golden 生成中 ...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             attn_implementation=ATTN_IMPL).cuda()
goldens = []
for cp in chat:
    ids = tokenizer(cp, return_tensors="pt").input_ids.cuda()
    out = model.generate(ids, do_sample=False, max_new_tokens=MAX_TOKENS,
                         pad_token_id=tokenizer.eos_token_id)
    goldens.append(out[0][ids.shape[1]:].tolist())
del model
torch.cuda.empty_cache()

# ---- nano-vllm greedy ----
print("== nano-vllm greedy 生成中 ...")
llm = LLM(MODEL, enforce_eager=True, max_model_len=2048)
outs = llm.generate(chat, SamplingParams(temperature=0, max_tokens=MAX_TOKENS), use_tqdm=False)
nvs = [o["token_ids"] for o in outs]

# ---- 对比 ----
rows, passed = [], 0
for i, (g, n) in enumerate(zip(goldens, nvs)):
    m = min(len(g), len(n))
    first = next((j for j in range(m) if g[j] != n[j]), m)
    identical = (first == m and len(g) == len(n))
    passed += first >= 128
    rows.append((i, PROMPTS[i][:22], first, min(first, 128), len(g), len(n), identical))

min_prefix = min(r[2] for r in rows)
verdict = "PASS" if min_prefix >= 128 else "FAIL"

lines = [
    "# LOGDIFF 报告(nano-vllm vs HF transformers golden)",
    "",
    f"- 模型:{MODEL}(BF16,无量化)| 采样:两侧 greedy | max_tokens:{MAX_TOKENS}",
    f"- 判定:**{verdict}**(最短共同前缀 {min_prefix} token,通过线 128)",
    "",
    "| # | prompt(截断) | 首异位置 | 共同前缀 | golden len | nano len | 完全一致 |",
    "| --- | --- | --- | --- | --- | --- | --- |",
]
for r in rows:
    lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[2]} | {r[4]} | {r[5]} | {'✅' if r[6] else '❌'} |")
lines += ["", f"通过条数(共同前缀>=128):{passed}/10"]
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[2:8]))
print(f"\n报告已写入 {os.path.abspath(REPORT)} | 判定:{verdict}")
