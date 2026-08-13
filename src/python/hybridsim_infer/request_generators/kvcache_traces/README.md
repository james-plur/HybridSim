# KV cache traces

公开 KV 前缀轨迹数据目录（`raw/` / `normalized/` / `samples/` / …）。

设计、数据源、归一化与 generator 用法见仓库文档：

**[docs/request_generation.md](../../../../../docs/request_generation.md)**

刷新归一化：

```bash
python3 tools/normalize_kvcache_traces.py
```
