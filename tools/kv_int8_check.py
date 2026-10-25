"""int8 KV 精度检查:同栈(nano-vllm)bf16 vs int8 的 greedy token 分叉。
用法:python tools/kv_int8_check.py bf16|int8  (各自进程跑,输出存 /tmp/kv_out_<mode>.json)
"""
import json, os, sys
from nanovllm import LLM, SamplingParams

mode = sys.argv[1]
MODEL = os.path.expanduser("~/huggingface/Qwen3-1.7B")
PROMPTS = [
    "用一句话介绍你自己", "写一首关于秋天的五言绝句", "推荐三本适合初学者的编程书籍,并说明理由",
    "解释什么是大语言模型的KV缓存", "Explain the difference between TCP and UDP in one paragraph.",
    "What causes the seasons on Earth?", "Write a Python function that checks if a number is prime.",
    "用Python实现快速排序,附注释", "Write a SQL query to find the second highest salary from an employees table.",
    "用bash写一个统计当前目录下文件总数的命令并解释",
]
kw = dict(enforce_eager=True, max_model_len=2048, kv_cache_dtype=mode)
llm = LLM(MODEL, **kw)
print(f"[{mode}] num_kvcache_blocks = {llm.model_runner.config.num_kvcache_blocks}")
tok = llm.tokenizer
chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
outs = llm.generate(chat, SamplingParams(temperature=0, max_tokens=256), use_tqdm=False)
json.dump([o["token_ids"] for o in outs], open(f"/tmp/kv_out_{mode}.json", "w"))
print(f"[{mode}] 输出已存 /tmp/kv_out_{mode}.json")
