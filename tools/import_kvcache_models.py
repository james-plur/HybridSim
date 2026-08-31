#!/usr/bin/env python3
"""Import GLM/DeepSeek/LLaMA/Kimi presets from elinx llm-mem-calculator data.js.

Usage:
  python tools/import_kvcache_models.py /path/to/data.js
  python tools/import_kvcache_models.py  # uses cached agent-tools copy if present
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

FAMILIES = frozenset({"DeepSeek", "GLM", "Kimi", "Llama"})
FORMULA_TO_ATTN = {
    "standard_gqa": "gqa",
    "mla": "mla",
    "dsa_mla": "dsa",
    "deepseek_v4_hybrid": "dsa",
}
FAMILY_DIR = {
    "DeepSeek": "deepseek",
    "GLM": "glm",
    "Kimi": "kimi",
    "Llama": "llama",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (
    REPO_ROOT
    / "src/python/hybridsim_infer/workload_generators/model_presets"
)


def _extract_models(data_js: Path) -> list[dict]:
    text = data_js.read_text(encoding="utf-8")
    marker = "const MODEL_DATA = "
    if marker not in text:
        raise ValueError(f"{data_js}: missing MODEL_DATA")
    tmp = Path("/tmp/_kvcache_model_data.js")
    tmp.write_text(text[text.index(marker) :] + "\nmodule.exports = MODEL_DATA;\n")
    out = subprocess.check_output(
        [
            "node",
            "-e",
            "const M=require('/tmp/_kvcache_model_data.js');"
            "const fam=new Set(['DeepSeek','GLM','Kimi','Llama']);"
            "console.log(JSON.stringify(M.models.filter(m=>fam.has(m.family))));",
        ],
        text=True,
    )
    return json.loads(out)


def _to_preset(m: dict) -> dict:
    f = m["fields"]
    w = m.get("weight_fields") or {}
    formula = m["formula"]
    family = m["family"]
    is_moe = bool(w.get("n_routed_experts"))
    moe_inter = int(w.get("moe_intermediate_size") or 0)
    dense_inter = int(w.get("intermediate_size") or 0)
    intermediate = moe_inter if (is_moe and moe_inter) else dense_inter
    share = 0
    if is_moe and int(w.get("n_shared_experts") or 0) > 0:
        share = int(w.get("shared_expert_intermediate_size") or moe_inter or 0)
    head_dim = int(
        f.get("head_dim")
        or ((f.get("qk_nope_head_dim") or 0) + (f.get("qk_rope_head_dim") or 0))
        or 128
    )
    kv_lora = int(f.get("kv_lora_rank") or 0)
    if formula == "deepseek_v4_hybrid" and kv_lora == 0:
        kv_lora = 512
    qk_nope = int(f.get("qk_nope_head_dim") or 0)
    qk_rope = int(f.get("qk_rope_head_dim") or 0)
    v_hd = int(f.get("v_head_dim") or 0)
    if formula == "deepseek_v4_hybrid":
        qk_nope = qk_nope or 128
        qk_rope = qk_rope or 64
        v_hd = v_hd or 128
    cfg = {
        "id": m["id"],
        "label": m["label"],
        "family": family.lower(),
        "source_url": m.get("source_url", ""),
        "kv_formula": formula,
        "attn_variant": FORMULA_TO_ATTN[formula],
        "ffn_activation": "silu",
        "dtype_bytes": 2,
        "num_layers": int(f["num_hidden_layers"]),
        "hidden_size": int(w.get("hidden_size") or 0),
        "intermediate_size": intermediate,
        "num_q_heads": int(
            w.get("num_attention_heads")
            or f.get("num_attention_heads")
            or f.get("num_key_value_heads")
            or 1
        ),
        "num_kv_heads": int(f.get("num_key_value_heads") or 1),
        "head_dim": head_dim,
        "q_lora_rank": int(w.get("q_lora_rank") or f.get("q_lora_rank") or 0),
        "kv_lora_rank": kv_lora,
        "qk_nope_head_dim": qk_nope,
        "qk_rope_head_dim": qk_rope,
        "v_head_dim": v_hd,
        "index_head_dim": int(f.get("index_head_dim") or 0),
        "index_n_heads": int(f.get("index_n_heads") or 0),
        "index_topk": int(f.get("index_topk") or 0),
        "index_topk_freq": int(f.get("index_topk_freq") or 0),
        "index_skip_topk_offset": int(f.get("index_skip_topk_offset") or 0),
        "sliding_window": int(f.get("sliding_window") or 0),
        "num_nextn_predict_layers": int(
            f.get("num_nextn_predict_layers") or f.get("mtp_transformer_layers") or 0
        ),
        "first_k_dense_replace": int(w.get("first_k_dense_replace") or 0),
        "is_moe": is_moe,
        "num_experts": int(w.get("n_routed_experts") or 1),
        "num_experts_per_tok": int(w.get("num_experts_per_tok") or 1),
        "share_expert_dim": share,
        "fused_add_norm": False,
    }
    if dense_inter and is_moe:
        cfg["dense_intermediate_size"] = dense_inter
    if "compress_ratios" in f:
        cfg["compress_ratios"] = list(f["compress_ratios"])
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "data_js",
        nargs="?",
        default="",
        help="Path to elinx llm-mem-calculator js/data.js",
    )
    args = ap.parse_args()
    data_js = Path(args.data_js) if args.data_js else None
    if data_js is None or not data_js.is_file():
        # Fall back to a previously fetched copy if present.
        candidates = sorted(
            Path.home().glob(
                ".cursor/projects/*/agent-tools/*26d50870*.txt"
            )
        )
        if not candidates:
            print("Provide path to data.js", file=sys.stderr)
            return 2
        data_js = candidates[-1]
    models = _extract_models(data_js)
    for fam in FAMILY_DIR.values():
        (OUT_ROOT / fam).mkdir(parents=True, exist_ok=True)
    for m in models:
        if m["family"] not in FAMILIES:
            continue
        cfg = _to_preset(m)
        out = OUT_ROOT / FAMILY_DIR[m["family"]] / f"{m['id']}.yaml"
        with out.open("w", encoding="utf-8") as fh:
            fh.write(
                f"# Auto-generated from elinx/llm-mem-calculator data.js ({m['id']})\n"
            )
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    print(f"imported {len(models)} presets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
