"""Register and construct inference frameworks by name."""

from __future__ import annotations

from typing import Any, Type

from hybridsim_infer.frameworks.base import InferenceFramework
from hybridsim_infer.frameworks.vllm import VllmFramework


class FrameworkFactory:
    """Simple name → class registry for replica schedulers / offline drivers."""

    _registry: dict[str, Type[InferenceFramework]] = {}

    @classmethod
    def register(cls, name: str, framework_cls: Type[InferenceFramework]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("framework name must be non-empty")
        cls._registry[key] = framework_cls

    @classmethod
    def create(cls, name: str = "vllm", **kwargs: Any) -> InferenceFramework:
        key = (name or "vllm").strip().lower()
        framework_cls = cls._registry.get(key)
        if framework_cls is None:
            known = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(f"unknown framework {name!r}; registered: {known}")
        return framework_cls(**kwargs)

    @classmethod
    def registered(cls) -> list[str]:
        return sorted(cls._registry)


# Built-in backends.
FrameworkFactory.register("vllm", VllmFramework)
