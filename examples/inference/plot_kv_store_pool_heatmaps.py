#!/usr/bin/env python3
"""Plot 5×5 KV Store sweep heatmaps: TTFT, TPS, prefix hit rate."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

_REPO = Path(__file__).resolve().parents[2]
_JSON = _REPO / "examples" / "inference" / "results" / "kv_store_pool_sweep.json"
_OUT = _REPO / "examples" / "inference" / "results"

_NOTO = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def _font() -> FontProperties | None:
    if _NOTO.exists():
        return FontProperties(fname=str(_NOTO))
    return None


def _fmt_bw_label(bw: float) -> str:
    if abs(bw - round(bw)) < 1e-9:
        return str(int(round(bw)))
    return f"{bw:g}"


def _axes_from_cells(cells: list[dict]) -> tuple[list[float], list[int]]:
    bws = sorted({float(c["kv_bandwidth_gbps"]) for c in cells if c.get("ok")})
    caps = sorted({int(c["kv_store_gb"]) for c in cells if c.get("ok")})
    return bws, caps


def load_grid(
    path: Path, field: str
) -> tuple[np.ndarray, list[float], list[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data["cells"]
    bandwidths, capacities = _axes_from_cells(cells)
    lookup: dict[tuple[float, int], float] = {}
    for cell in cells:
        if not cell.get("ok"):
            continue
        lookup[(float(cell["kv_bandwidth_gbps"]), int(cell["kv_store_gb"]))] = float(
            cell[field]
        )
    grid = np.full((len(bandwidths), len(capacities)), np.nan)
    for i, bw in enumerate(bandwidths):
        for j, cap in enumerate(capacities):
            grid[i, j] = lookup.get((bw, cap), np.nan)
    return grid, bandwidths, capacities


def plot_one(
    grid: np.ndarray,
    *,
    bandwidths: list[float],
    capacities: list[int],
    title: str,
    cbar_label: str,
    cmap: str,
    fmt,
    outfile: Path,
    font: FontProperties | None,
) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
    im = ax.imshow(grid, origin="lower", cmap=cmap, aspect="equal")
    ax.set_xticks(range(len(capacities)), [str(c) for c in capacities])
    ax.set_yticks(range(len(bandwidths)), [_fmt_bw_label(b) for b in bandwidths])
    ax.set_xlabel("Store 容量 (GB)", fontproperties=font)
    ax.set_ylabel("拉取带宽 (Gbps)", fontproperties=font)
    ax.set_title(title, fontproperties=font)
    if font is not None:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontproperties(font)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j,
                i,
                fmt(grid[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                fontproperties=font,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontproperties=font)
    if font is not None:
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(font)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outfile}")

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_JSON,
        help="Sweep JSON (default: kv_store_pool_sweep.json)",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="",
        help="Filename infix, e.g. flops8x → heatmap_ttft_flops8x.png",
    )
    args = parser.parse_args()
    infix = f"_{args.out_prefix}" if args.out_prefix else ""
    font = _font()
    def ttft_fmt(v: float) -> str:
        if v != v:  # NaN
            return ""
        if abs(v) < 10:
            return f"{v:.3f}"
        if abs(v) < 100:
            return f"{v:.1f}"
        return f"{v:.0f}"
    specs = (
        (
            "mean_ttft_s",
            "Mean TTFT",
            "TTFT (s)",
            "YlOrRd",
            ttft_fmt,
            _OUT / f"heatmap_ttft{infix}.png",
        ),
        (
            "tps",
            "系统 TPS",
            "prefill tokens / s",
            "YlGn",
            lambda v: f"{v:.0f}",
            _OUT / f"heatmap_tps{infix}.png",
        ),
        (
            "hit_rate",
            "Prefix cache 命中率",
            "hit tokens / prefill tokens",
            "YlGn",
            lambda v: f"{v * 100:.1f}%",
            _OUT / f"heatmap_hit_rate{infix}.png",
        ),
    )
    for field, title, cbar, cmap, fmt, out in specs:
        grid, bandwidths, capacities = load_grid(args.input, field)
        plot_one(
            grid,
            bandwidths=bandwidths,
            capacities=capacities,
            title=title,
            cbar_label=cbar,
            cmap=cmap,
            fmt=fmt,
            outfile=out,
            font=font,
        )


if __name__ == "__main__":
    main()
