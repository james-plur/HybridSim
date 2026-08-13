#!/usr/bin/env python3
"""Normalize public kvcache-simulator traces into Mooncake-style JSONL + catalog.

Public sources publish remapped ``hash_ids`` + lengths, not raw ``input_ids``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "python"
    / "hybridsim_infer"
    / "request_generators"
    / "kvcache_traces"
)
RAW = ROOT / "raw"
NORM = ROOT / "normalized"
META = ROOT / "meta"


def _write_with_block_size(src: Path, dest: Path, block_size: int) -> int:
    n = 0
    with src.open() as reader, dest.open("w") as writer:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row.setdefault("block_size", block_size)
            writer.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += 1
    return n


def flatten_weka_sessions(src_dir: Path, dest: Path) -> dict[str, int]:
    """Remap per-session local hash ids into a global int namespace."""
    remapper: dict[tuple[str, int], int] = {}
    next_id = 0

    def map_id(trace_id: str, value: int) -> int:
        nonlocal next_id
        key = (trace_id, int(value))
        assigned = remapper.get(key)
        if assigned is None:
            assigned = next_id
            remapper[key] = assigned
            next_id += 1
        return assigned

    n_req = 0
    n_tr = 0
    base_ts = 0.0
    with dest.open("w") as writer:
        for path in sorted(src_dir.glob("*.json")):
            n_tr += 1
            session = json.loads(path.read_text())
            block_size = int(session.get("block_size") or 64)
            trace_id = str(session.get("id") or path.stem)
            requests = list(session.get("requests") or [])
            max_t = 0.0
            for req in requests:
                hashes = req.get("hash_ids") or []
                if not hashes:
                    continue
                t = float(req.get("t") or 0.0)
                max_t = max(max_t, t)
                row = {
                    "id": f"{trace_id}:{n_req}",
                    "trace_id": trace_id,
                    "timestamp": base_ts + t,
                    "block_size": block_size,
                    "input_length": int(req.get("in") or req.get("input_length") or 0),
                    "output_length": int(req.get("out") or req.get("output_length") or 0),
                    "hash_ids": [map_id(trace_id, h) for h in hashes],
                    "hash_id_scope": "local_remapped",
                    "model": req.get("model"),
                }
                writer.write(json.dumps(row, separators=(",", ":")) + "\n")
                n_req += 1
            base_ts += max_t + 1.0
    return {
        "traces": n_tr,
        "requests": n_req,
        "unique_blocks": len(remapper),
        "bytes": dest.stat().st_size,
    }


def main() -> None:
    NORM.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    stats: dict[str, object] = {}
    for src_name, block_size, dest_name in [
        ("mooncake_trace.jsonl", 512, "mooncake_fast25.jsonl"),
        ("qwen_traceA_blksz_16.jsonl", 16, "qwen_bailian_traceA.jsonl"),
        ("qwen_traceB_blksz_16.jsonl", 16, "qwen_bailian_traceB.jsonl"),
        ("qwen_coder_blksz_16.jsonl", 16, "qwen_bailian_coder.jsonl"),
        ("qwen_thinking_blksz_16.jsonl", 16, "qwen_bailian_thinking.jsonl"),
    ]:
        src = RAW / src_name
        dest = NORM / dest_name
        if not src.exists() or src.stat().st_size < 1000:
            stats[dest_name] = {"status": "missing", "path": str(src)}
            continue
        n = _write_with_block_size(src, dest, block_size)
        stats[dest_name] = {"requests": n, "bytes": dest.stat().st_size, "block_size": block_size}

    weka_src = RAW / "kv_cache_tester"
    weka_dest = NORM / "weka_claude_code_kv_cache_tester.jsonl"
    if weka_src.is_dir():
        stats[weka_dest.name] = flatten_weka_sessions(weka_src, weka_dest)

    catalog = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Public kvcache-simulator traces provide remapped hash_ids + lengths, "
            "NOT raw input token ids (privacy). To obtain both tokens and a hash "
            "chain, convert SGLang Finish logs via "
            "kvcache-blog/scripts/sglang-log-to-kvcache-trace.py (extend it to keep "
            "input_ids if hybridsim needs both)."
        ),
        "normalized_stats": stats,
        "official_sources": [
            {
                "id": "mooncake_fast25",
                "kind": "hash",
                "block_size": 512,
                "local": "normalized/mooncake_fast25.jsonl",
                "url": "https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/arxiv-trace",
                "has_input_ids": False,
            },
            {
                "id": "qwen_bailian_traceA",
                "kind": "hash",
                "block_size": 16,
                "local": "normalized/qwen_bailian_traceA.jsonl",
                "url": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
                "has_input_ids": False,
            },
            {
                "id": "qwen_bailian_traceB",
                "kind": "hash",
                "block_size": 16,
                "local": "normalized/qwen_bailian_traceB.jsonl",
                "url": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
                "has_input_ids": False,
            },
            {
                "id": "qwen_bailian_coder",
                "kind": "hash",
                "block_size": 16,
                "local": "normalized/qwen_bailian_coder.jsonl",
                "url": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
                "has_input_ids": False,
            },
            {
                "id": "qwen_bailian_thinking",
                "kind": "hash",
                "block_size": 16,
                "local": "normalized/qwen_bailian_thinking.jsonl",
                "url": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
                "has_input_ids": False,
            },
            {
                "id": "weka_claude_code_kv_cache_tester",
                "kind": "hash",
                "block_size": 64,
                "local": "normalized/weka_claude_code_kv_cache_tester.jsonl",
                "url": "https://github.com/callanjfox/kv-cache-tester",
                "has_input_ids": False,
                "note": "Same corpus family as SemiAnalysis Weka; local hash ids remapped globally.",
            },
            {
                "id": "ragpulse",
                "kind": "hash",
                "block_size": 512,
                "url": "https://huggingface.co/datasets/flashserve/RAGPulse",
                "has_input_ids": False,
                "status": "download_blocked_hf_proxy_403",
            },
            {
                "id": "semianalysis_weka_no_subagents",
                "kind": "hash",
                "block_size": 64,
                "url": "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226",
                "has_input_ids": False,
                "status": "download_blocked_hf_proxy_403",
                "fallback": "weka_claude_code_kv_cache_tester",
            },
            {
                "id": "semianalysis_weka_with_subagents_256k",
                "kind": "hash",
                "block_size": 64,
                "url": "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-with-subagents-052726-256k",
                "has_input_ids": False,
                "status": "download_blocked_hf_proxy_403",
            },
            {
                "id": "lmcache_agentic",
                "kind": "agent_text",
                "block_size": 64,
                "url": "https://huggingface.co/datasets/zeelHz/lmcache-agentic-traces",
                "has_input_ids": False,
                "status": "download_blocked_hf_proxy_403",
            },
            {
                "id": "exgentic_agent",
                "kind": "agent_text",
                "block_size": 64,
                "url": "https://huggingface.co/datasets/Exgentic/agent-llm-traces",
                "has_input_ids": False,
                "status": "download_blocked_hf_proxy_403",
            },
        ],
        "token_plus_hash_path": {
            "tool": "https://github.com/kvcache-ai/kvcache-blog/blob/main/scripts/sglang-log-to-kvcache-trace.py",
            "requires": "SGLang Finish logs containing input_ids=[...]",
            "note": (
                "Official converter emits hash_ids only. Keep input_ids in the "
                "JSONL if hybridsim Store should hash via vLLM chain AND retain "
                "real tokens for compute."
            ),
        },
    }
    (META / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    print(f"catalog -> {META / 'catalog.json'}")


if __name__ == "__main__":
    main()
