"""Weight / activation shape helpers for TP sharding."""

from __future__ import annotations


class Shape:
    """Mutable dimension list; ``split`` shards one axis across ``parts`` ranks."""

    def __init__(self, dims: list[int] | tuple[int, ...]) -> None:
        self.dims = [int(d) for d in dims]

    def clone(self) -> Shape:
        return Shape(self.dims)

    def split(self, dim: int, parts: int) -> Shape:
        """In-place column/row parallel split; returns ``self`` for chaining."""
        parts = max(1, int(parts))
        dim = int(dim)
        if dim < 0 or dim >= len(self.dims):
            raise IndexError(f"split dim {dim} out of range for {self.dims}")
        self.dims[dim] = max(1, int(self.dims[dim]) // parts)
        return self

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(self.dims)

    def __repr__(self) -> str:
        return f"Shape({self.dims})"
