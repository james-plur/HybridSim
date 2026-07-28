"""Wrap a pre-built request list as a RequestGenerator."""

from __future__ import annotations

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.request_generators.base import RequestGenerator


class ListRequestGenerator(RequestGenerator):
    """Return a fixed list of ``InferenceRequest`` (demos / tests / hand-built traces)."""

    def __init__(self, requests: list[InferenceRequest]) -> None:
        self._requests = list(requests)

    def generate(self) -> list[InferenceRequest]:
        return list(self._requests)
