import os
import sys
import time
from random import randint, seed
import torch
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    enforce_eager = len(sys.argv) > 2 and sys.argv[2] == "eager"
    gpu_util = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    max_input_len = 512
    max_ouput_len = 512

    path = os.path.expanduser("~/huggingface/Qwen3-1.7B/")
    llm = LLM(path, enforce_eager=enforce_eager, max_model_len=2048,
              gpu_memory_utilization=gpu_util)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    print(f"显存已分配(张量真用): {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"显存已预留(池子):     {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"显存峰值分配:         {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
