"""TPOT 基准:逐 batch 测 decode 单步延迟(median of N)。运行:python tools/tpot_bench.py"""
import os, time
from nanovllm import LLM, SamplingParams

BATCHES = [1, 2, 4, 8, 32]
REPEAT = 3
TOKENS = 256

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-1.7B"), enforce_eager=False, max_model_len=2048)
llm.generate(["warmup"], SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)

print(f"| bs | TPOT ms (median of {REPEAT}) | 聚合 tok/s |")
print("| --- | --- | --- |")
for bs in BATCHES:
    prompts = [[1000 + i] * 64 for i in range(bs)]
    sp = SamplingParams(temperature=0, max_tokens=TOKENS, ignore_eos=True)
    tpots = []
    for _ in range(REPEAT):
        t = time.time()
        llm.generate(prompts, sp, use_tqdm=False)
        tpots.append((time.time() - t) / TOKENS * 1000)
    tpots.sort()
    med = tpots[len(tpots)//2]
    print(f"| {bs} | {med:.2f} | {bs*1000/med:.0f} |")
