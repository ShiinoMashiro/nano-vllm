# LOGDIFF 报告(nano-vllm vs HF transformers golden)

- 模型:/home/yaojun/huggingface/Qwen3-1.7B(BF16,无量化)| 采样:两侧 greedy | max_tokens:256
- 判定:**FAIL**(最短共同前缀 4 token,通过线 128)

| # | prompt(截断) | 首异位置 | 共同前缀 | golden len | nano len | 完全一致 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 用一句话介绍你自己 | 39 | 39 | 256 | 216 | ❌ |
| 1 | 写一首关于秋天的五言绝句 | 76 | 76 | 256 | 256 | ❌ |
| 2 | 推荐三本适合初学者的编程书籍,并说明理由 | 51 | 51 | 256 | 256 | ❌ |
| 3 | 解释什么是大语言模型的KV缓存 | 149 | 149 | 256 | 256 | ❌ |
| 4 | Explain the difference | 4 | 4 | 256 | 256 | ❌ |
| 5 | What causes the season | 108 | 108 | 256 | 256 | ❌ |
| 6 | Write a Python functio | 96 | 96 | 256 | 256 | ❌ |
| 7 | 用Python实现快速排序,附注释 | 43 | 43 | 256 | 256 | ❌ |
| 8 | Write a SQL query to f | 65 | 65 | 256 | 256 | ❌ |
| 9 | 用bash写一个统计当前目录下文件总数的命令 | 156 | 156 | 256 | 256 | ❌ |

通过条数(共同前缀>=128):2/10
