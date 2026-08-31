"""Shared model enums used by presets, infer op-level, and KV volume formulas."""

from __future__ import annotations

from enum import Enum


class AttnVariant(Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    MLA = "mla"
    CSA = "csa"
    HSA = "hsa"
    DSA = "dsa"


_ATTN_IMPLEMENTED = frozenset(
    {
        AttnVariant.MHA,
        AttnVariant.GQA,
        AttnVariant.MQA,
        AttnVariant.MLA,
        AttnVariant.DSA,
    }
)
_ATTN_STUBBED = frozenset({AttnVariant.CSA, AttnVariant.HSA})


def ensure_attn_variant_supported(variant: AttnVariant) -> None:
    if variant in _ATTN_STUBBED:
        raise NotImplementedError(
            f"Attention variant {variant.value!r} is registered as an extension "
            "point but not implemented yet (supported: mha, gqa, mqa, mla, dsa)"
        )
    if variant not in _ATTN_IMPLEMENTED:
        raise ValueError(f"Unknown attention variant: {variant!r}")


class FfnActivation(Enum):
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"
    RELU = "relu"

