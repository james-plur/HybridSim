"""Torch-like Module for nested mock model construction."""

from __future__ import annotations

from typing import Any, Iterator


class Module:
    """Minimal ``nn.Module`` stand-in: children register via attribute assignment."""

    def __init__(self) -> None:
        object.__setattr__(self, "_children", {})

    def __setattr__(self, name: str, value: Any) -> None:
        children = object.__getattribute__(self, "__dict__").get("_children")
        if children is not None and isinstance(value, Module) and name != "_children":
            children[name] = value
        object.__setattr__(self, name, value)

    def named_children(self) -> Iterator[tuple[str, Module]]:
        children: dict[str, Module] = object.__getattribute__(self, "_children")
        yield from children.items()

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        yield prefix or type(self).__name__, self
        for name, child in self.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(child_prefix)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward is not implemented")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
