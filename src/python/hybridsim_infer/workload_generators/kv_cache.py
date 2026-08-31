"""KV cache byte formulas aligned with elinx/llm-mem-calculator ``calc.js``.

Callers pass a ``ModelConfig`` or a ``model_presets`` id (e.g. ``llama-3.1-8b``).
Shape fields (layers, heads, MLA ranks, indexer, compress ratios) come from the
preset YAML — users only pick the model kind.
"""

from __future__ import annotations

from hybridsim_infer.workload_generators.configs import ModelConfig

KV_FORMULA_GQA = "standard_gqa"
KV_FORMULA_MLA = "mla"
KV_FORMULA_DSA_MLA = "dsa_mla"
KV_FORMULA_V4_HYBRID = "deepseek_v4_hybrid"

ModelRef = ModelConfig | str


def resolve_model(model: ModelRef) -> ModelConfig:
    """Return ``ModelConfig``; load from :func:`load_preset` when given an id."""
    if isinstance(model, str):
        from hybridsim_infer.workload_generators.model_presets import load_preset

        return load_preset(model)
    return model


def resolved_kv_formula(model: ModelRef) -> str:
    model = resolve_model(model)
    raw = (getattr(model, "kv_formula", None) or "").strip().lower()
    if raw:
        return raw
    variant = model.resolved_attn_variant().value
    if variant == "dsa":
        return KV_FORMULA_DSA_MLA
    if variant == "mla":
        return KV_FORMULA_MLA
    return KV_FORMULA_GQA


def _prec_b(model: ModelConfig) -> float:
    return float(max(0.0, int(model.dtype_bytes)))


def _idx_b(model: ModelConfig) -> float:
    idx = int(getattr(model, "index_dtype_bytes", 0) or 0)
    if idx > 0:
        return float(idx)
    return _prec_b(model)


def _include_draft(model: ModelConfig, include_draft: bool | None) -> bool:
    if include_draft is not None:
        return bool(include_draft)
    return bool(getattr(model, "include_draft_kv", False))


def _num_indexer_layers(model: ModelConfig) -> int:
    layers = max(0, int(model.num_layers))
    freq = int(getattr(model, "index_topk_freq", 0) or 0)
    if freq <= 0:
        return layers
    offset = int(getattr(model, "index_skip_topk_offset", 0) or 0)
    return offset + (layers - offset) // freq


def cache_bytes(
    model: ModelRef,
    num_tokens: int,
    *,
    include_draft: bool | None = None,
) -> float:
    """Total KV (+ indexer) bytes for ``num_tokens`` tokens (one sequence).

    ``model`` is a ``ModelConfig`` or a preset id such as ``llama-3.1-8b``.
    """
    model = resolve_model(model)
    tokens = max(0, int(num_tokens))
    formula = resolved_kv_formula(model)
    prec = _prec_b(model)
    idx_prec = _idx_b(model)
    draft = _include_draft(model, include_draft)

    if formula == KV_FORMULA_GQA:
        layers = max(0, int(model.num_layers))
        n_kv = max(1, int(model.num_kv_heads))
        hd = max(1, int(model.head_dim))
        elements = 2 * layers * n_kv * hd * tokens
        nbytes = elements * prec
        if draft:
            draft_layers = int(getattr(model, "num_nextn_predict_layers", 0) or 0)
            if draft_layers > 0:
                nbytes += 2 * draft_layers * n_kv * hd * tokens * prec
        return float(nbytes)

    if formula == KV_FORMULA_MLA:
        layers = max(0, int(model.num_layers))
        kv_lora = max(0, int(model.kv_lora_rank))
        rope = max(0, int(model.qk_rope_head_dim))
        elements = layers * (kv_lora + rope) * tokens
        nbytes = elements * prec
        if draft:
            draft_layers = int(getattr(model, "num_nextn_predict_layers", 0) or 0)
            if draft_layers > 0:
                nbytes += draft_layers * (kv_lora + rope) * tokens * prec
        return float(nbytes)

    if formula == KV_FORMULA_DSA_MLA:
        layers = max(0, int(model.num_layers))
        kv_lora = max(0, int(model.kv_lora_rank))
        rope = max(0, int(model.qk_rope_head_dim))
        elements = layers * (kv_lora + rope) * tokens
        nbytes = elements * prec
        idx_hd = max(0, int(getattr(model, "index_head_dim", 0) or 0))
        n_idx = _num_indexer_layers(model)
        nbytes += n_idx * idx_hd * tokens * idx_prec
        if draft:
            draft_layers = int(getattr(model, "num_nextn_predict_layers", 0) or 0)
            if draft_layers > 0:
                nbytes += draft_layers * (kv_lora + rope) * tokens * prec
        return float(nbytes)

    if formula == KV_FORMULA_V4_HYBRID:
        ratios = list(getattr(model, "compress_ratios", None) or [])
        if not ratios:
            raise ValueError(
                "deepseek_v4_hybrid requires ModelConfig.compress_ratios"
            )
        sw = max(0, int(getattr(model, "sliding_window", 0) or 0))
        hd = max(1, int(model.head_dim))
        idx_hd = max(0, int(getattr(model, "index_head_dim", 0) or 0))
        total_layers = max(0, int(model.num_layers))
        sliding_elements = total_layers * sw * hd
        compressed_elements = 0
        ratio4_layers = 0
        for r in ratios:
            r = int(r)
            if r > 0:
                compressed_elements += (tokens // r) * hd
            if r == 4:
                ratio4_layers += 1
        kv_elements = sliding_elements + compressed_elements
        nbytes = kv_elements * prec
        idx_elements = ratio4_layers * (tokens // 4) * idx_hd
        nbytes += idx_elements * idx_prec
        if draft:
            ratio0 = sum(1 for r in ratios if int(r) == 0)
            nbytes += ratio0 * sw * hd * prec
        return float(nbytes)

    raise ValueError(f"Unknown kv_formula: {formula!r}")


def bytes_per_token(
    model: ModelRef,
    *,
    num_tokens: int = 1,
    include_draft: bool | None = None,
) -> float:
    """Average bytes per token for a sequence of length ``num_tokens``.

    For GQA/MLA/DSA this equals the T=1 full-layer volume. For V4 hybrid the
    per-token average depends on ``num_tokens`` (floor compression).
    """
    t = max(1, int(num_tokens))
    return cache_bytes(model, t, include_draft=include_draft) / float(t)
