"""prefix cache 收益量化:共享前缀 workload vs 互不相同 workload。

mode a:32 条互不相同的 768-token prompt(max_tokens=1,墙钟≈纯 prefill)
mode b:先用 1 条 768-token 请求填充缓存,再计时 31 条"共享前缀+64 token 尾巴"请求
       (模拟多轮对话/共享系统提示词场景:跨请求缓存命中)

运行:python tools/prefix_bench.py a|b
"""
import os
import sys
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams

seed(0)
mode = sys.argv[1]
NUM = 32
PREFIX_LEN, SUFFIX_LEN = 768, 64

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-1.7B"),
          enforce_eager=False, max_model_len=2048)
sp = SamplingParams(temperature=0.6, max_tokens=1)   # 1 token:墙钟≈prefill

if mode == "a":
    prompts = [[randint(1000, 14000) for _ in range(PREFIX_LEN)] for _ in range(NUM)]
    llm.generate(["warm"], SamplingParams(max_tokens=1))
    t = time.time()
    llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t
    total = sum(len(p) for p in prompts)
    print(f"mode=a 无共享 | prefill tokens={total} | wall={dt:.2f}s | prefill速率={total/dt:.0f} tok/s")
elif mode == "b":
    shared = [randint(1000, 14000) for _ in range(PREFIX_LEN)]
    llm.generate([shared], sp, use_tqdm=False)          # 第 1 条:填充缓存
    prompts = [shared + [randint(1000, 14000) for _ in range(SUFFIX_LEN)] for _ in range(NUM - 1)]
    t = time.time()
    llm.generate(prompts, sp, use_tqdm=False)           # 计时这 31 条
    dt = time.time() - t
    total = sum(len(p) for p in prompts)
    eff = (NUM - 1) * SUFFIX_LEN                        # 若前缀全命中,实际只需计算的量
    print(f"mode=b 共享前缀 | 提交 tokens={total} | 命中后应算≈{eff} | wall={dt:.2f}s | 表观速率={total/dt:.0f} tok/s")
