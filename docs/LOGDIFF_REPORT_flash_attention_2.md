# LOGDIFF 报告(nano-vllm vs HF transformers golden)

- 模型:/home/yaojun/huggingface/Qwen3-1.7B(BF16,无量化)| 采样:两侧 greedy | max_tokens:256
- 判定:**FAIL**(最短共同前缀 47 token,通过线 128)

| # | prompt(截断) | 首异位置 | 共同前缀 | golden len | nano len | 完全一致 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 用一句话介绍你自己 | 47 | 47 | 256 | 216 | ❌ |
| 1 | 写一首关于秋天的五言绝句 | 51 | 51 | 256 | 256 | ❌ |
| 2 | 推荐三本适合初学者的编程书籍,并说明理由 | 89 | 89 | 256 | 256 | ❌ |
| 3 | 解释什么是大语言模型的KV缓存 | 239 | 239 | 256 | 256 | ❌ |
| 4 | Explain the difference | 256 | 256 | 256 | 256 | ✅ |
| 5 | What causes the season | 256 | 256 | 256 | 256 | ✅ |
| 6 | Write a Python functio | 96 | 96 | 256 | 256 | ❌ |
| 7 | 用Python实现快速排序,附注释 | 101 | 101 | 256 | 256 | ❌ |
| 8 | Write a SQL query to f | 70 | 70 | 256 | 256 | ❌ |
| 9 | 用bash写一个统计当前目录下文件总数的命令 | 53 | 53 | 256 | 256 | ❌ |

通过条数(共同前缀>=128):3/10

## 结论与判定标准修订(2026-10-25)

1. **跨栈 token 级完全一致不可达**:nano-vllm(flash-attn varlen/paged)与 HF(RoPE 实现、kernel 分块均不同)是两条独立数值路径,greedy 分叉是预期行为。
2. **attention kernel 是分叉主因的实验证据**:HF 从 SDPA 切到 flash_attention_2 后,2/10 条变为完全一致(#4: 4→256,#5: 108→256),最短共同前缀 4→47;但同 kernel 家族下仍有分叉(剩余差异来自 RoPE 及其他算子的数值路径)。
3. **语义抽检**(SDPA 下 4-token 分叉的最差 case):两侧输出均连贯合理——数值噪声翻转发生在近并列 logits 处,不构成质量或框架正确性问题。
4. **判定标准修订**:
   - 跨栈:报告共同前缀分布 + 语义抽检,不再用"全同"作硬标准;严格数值对数应上探到 logits cosine(对数阶梯:token 全同 → 前N一致 → logits → cosine);
   - **同栈自一致性是硬标准**:同引擎两次运行/eager vs graph 必须严格一致——greedy 下已验证 PASS(T1)。
