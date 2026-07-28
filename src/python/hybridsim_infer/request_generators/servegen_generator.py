"""ServeGen-backed RequestGenerator (LANGUAGE → InferenceRequest).

Optional dependency: install ServeGen from
https://github.com/alibaba/ServeGen (``pip install -e <clone>``).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from hybridsim_infer.request import InferenceRequest
from hybridsim_infer.request_generators.base import RequestGenerator

_SERVEGEN_INSTALL_HINT = (
    "ServeGen is required for ServeGenRequestGenerator. Install with: "
    "pip install -e <path-to-alibaba/ServeGen> "
    "(see https://github.com/alibaba/ServeGen)"
)


def _import_servegen() -> tuple[Any, Any, Any, Any]:
    try:
        from servegen import Category, ClientPool, generate_workload
        from servegen.utils import get_constant_rate_fn
    except ImportError as exc:
        raise ImportError(_SERVEGEN_INSTALL_HINT) from exc
    return Category, ClientPool, generate_workload, get_constant_rate_fn


def _servegen_workdir() -> Path:
    """Directory that contains ServeGen's ``data/`` tree (ClientPool uses a relative path)."""
    import servegen

    pkg = Path(servegen.__file__).resolve().parent
    for root in (pkg.parent, pkg, Path.cwd()):
        if (root / "data" / "language").is_dir():
            return root
    raise FileNotFoundError(
        "ServeGen data/ directory not found next to the package or cwd. "
        "Install with an editable clone: pip install -e <path-to-alibaba/ServeGen>"
    )


@contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def map_servegen_request(
    raw: Any,
    *,
    id_offset: int = 0,
    time_offset: float = 0.0,
) -> InferenceRequest:
    """Map a ServeGen ``Request`` (or duck-typed object) to ``InferenceRequest``.

    Expects LANGUAGE fields: ``data['input_tokens']``, ``data['output_tokens']``.
    """
    data = getattr(raw, "data", None)
    if not isinstance(data, dict):
        raise ValueError(f"ServeGen request missing data dict: {raw!r}")
    if "input_tokens" not in data or "output_tokens" not in data:
        raise ValueError(
            "ServeGen LANGUAGE mapping requires data['input_tokens'] and "
            f"data['output_tokens']; got keys={sorted(data.keys())}"
        )
    rid = getattr(raw, "request_id", None)
    if rid is None:
        raise ValueError("ServeGen request missing request_id")
    ts = getattr(raw, "timestamp", None)
    if ts is None:
        raise ValueError("ServeGen request missing timestamp")
    return InferenceRequest(
        request_id=int(id_offset) + int(rid),
        arrived_at=float(ts) + float(time_offset),
        num_prefill_tokens=int(data["input_tokens"]),
        num_decode_tokens=int(data["output_tokens"]),
    )


class ServeGenRequestGenerator(RequestGenerator):
    """Call ServeGen ``generate_workload`` then map to ``InferenceRequest``.

    First version supports ``category='language'`` only.
    """

    def __init__(
        self,
        *,
        category: str = "language",
        model: str = "m-small",
        duration: int = 60,
        rate: float = 5.0,
        rate_fn: Optional[dict[int, float]] = None,
        seed: Optional[int] = 0,
        max_requests: Optional[int] = None,
        time_offset: float = 0.0,
        id_offset: int = 0,
    ) -> None:
        Category, ClientPool, generate_workload, get_constant_rate_fn = (
            _import_servegen()
        )
        cat = (category or "language").lower().strip()
        if cat != "language":
            raise ValueError(
                "ServeGenRequestGenerator v1 only supports category='language' "
                f"(got {category!r})"
            )
        if duration <= 0:
            raise ValueError("duration must be positive")
        self._Category = Category
        self._ClientPool = ClientPool
        self._generate_workload = generate_workload
        self._get_constant_rate_fn = get_constant_rate_fn
        self._model = model
        self._duration = int(duration)
        self._rate = float(rate)
        self._rate_fn = dict(rate_fn) if rate_fn is not None else None
        self._seed = seed
        self._max_requests = max_requests
        self._time_offset = float(time_offset)
        self._id_offset = int(id_offset)

    def generate(self) -> list[InferenceRequest]:
        # ClientPool loads ``data/{category}/{model}`` relative to cwd.
        with _chdir(_servegen_workdir()):
            pool = self._ClientPool(self._Category.LANGUAGE, self._model)
            view = pool.span(0, self._duration)
            rate_fn = (
                self._rate_fn
                if self._rate_fn is not None
                else self._get_constant_rate_fn(view, self._rate)
            )
            if not rate_fn:
                return []
            raw_requests = self._generate_workload(
                view,
                rate_fn,
                duration=self._duration,
                seed=self._seed,
            )
        if self._max_requests is not None:
            raw_requests = raw_requests[: int(self._max_requests)]
        return [
            map_servegen_request(
                r,
                id_offset=self._id_offset,
                time_offset=self._time_offset,
            )
            for r in raw_requests
        ]
