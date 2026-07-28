"""Batch duration predictors used by timeout-kernel workload generators."""

from __future__ import annotations

from typing import Protocol

from hybridsim_infer.schedule_types import DecodeChunk, PrefillChunk, ScheduleBatch


class BatchDurationPredictor(Protocol):
    def predict(self, schedule_batch: ScheduleBatch) -> float:
        """Return simulated batch duration in seconds."""


class FixedDurationPredictor:
    def __init__(self, duration_s: float = 0.05) -> None:
        self.duration_s = float(duration_s)

    def predict(self, schedule_batch: ScheduleBatch) -> float:
        return self.duration_s


class TokenProportionalPredictor:
    """duration = base + prefill_tokens * p + decode_tokens * d."""

    def __init__(
        self,
        *,
        prefill_s_per_token: float = 1e-4,
        decode_s_per_token: float = 1e-3,
        base_s: float = 0.0,
    ) -> None:
        self.prefill_s_per_token = float(prefill_s_per_token)
        self.decode_s_per_token = float(decode_s_per_token)
        self.base_s = float(base_s)

    def predict(self, schedule_batch: ScheduleBatch) -> float:
        prefill = 0
        decode = 0
        for chunk in schedule_batch.chunks:
            if isinstance(chunk, PrefillChunk):
                prefill += int(chunk.num_tokens)
            elif isinstance(chunk, DecodeChunk):
                decode += int(chunk.num_tokens)
            else:
                decode += int(getattr(chunk, "num_tokens", 0))
        if prefill == 0 and decode == 0:
            total = sum(int(v) for v in schedule_batch.tokens_per_request.values())
            decode = total
        return (
            self.base_s
            + prefill * self.prefill_s_per_token
            + decode * self.decode_s_per_token
        )


def make_predictor(
    *,
    duration_mode: str = "fixed",
    dummy_exec_s: float = 0.05,
    prefill_s_per_token: float = 1e-4,
    decode_s_per_token: float = 1e-3,
    base_s: float = 0.0,
) -> FixedDurationPredictor | TokenProportionalPredictor:
    mode = (duration_mode or "fixed").lower()
    if mode == "token_proportional":
        return TokenProportionalPredictor(
            prefill_s_per_token=prefill_s_per_token,
            decode_s_per_token=decode_s_per_token,
            base_s=base_s,
        )
    return FixedDurationPredictor(duration_s=dummy_exec_s)
