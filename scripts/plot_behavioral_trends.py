#!/usr/bin/env python3
"""Down-sampled behavioral metrics plots for large runs.

Usage:
    python scripts/plot_behavioral_trends.py [--logs-dir experiments/logs]

The script streams the fields needed from metrics.json, applies stride-based
down-sampling plus a rolling average, and writes the resulting figures into the
chosen logging directory. Designed to handle very large metrics files without
loading them fully into memory.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Iterator, Optional

import matplotlib

matplotlib.use("Agg")  # ensure headless rendering before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    import ijson  # type: ignore
except ImportError as exc:  # pragma: no cover - defensive guardrail
    raise SystemExit(
        "Missing dependency: install ijson (e.g. `pip install ijson`)."
    ) from exc


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("experiments/logs"),
        help="Directory containing metrics.json (default: experiments/logs)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5000,
        help="Keep every Nth sample when streaming metrics (default: 5000).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Window size for rolling average smoothing (default: 5 samples).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def stream_series(path: Path, key: str, stride: int) -> Iterator[float]:
    """Yield every `stride`th numeric entry for `key` from metrics.json."""
    with path.open("rb") as fh:
        parser = ijson.parse(fh)
        count = 0
        target_prefix = f"{key}.item"
        for prefix, event, value in parser:
            if prefix == target_prefix and event == "number":
                if stride <= 1 or count % stride == 0:
                    yield float(value)
                count += 1
            elif prefix == key and event == "end_array":
                break


def rolling_mean(values: Iterable[float], window: int) -> np.ndarray:
    """Compute rolling mean with bounded window."""
    if window <= 1:
        return np.fromiter(values, dtype=float)

    buffer: Deque[float] = deque()
    total = 0.0
    averaged = []
    for val in values:
        buffer.append(val)
        total += val
        if len(buffer) > window:
            total -= buffer.popleft()
        averaged.append(total / len(buffer))
    return np.array(averaged, dtype=float)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    metrics_path = args.logs_dir / "metrics.json"

    if not metrics_path.exists():
        print(f"No metrics.json found at {metrics_path}", file=sys.stderr)
        return 1

    stride = max(1, args.stride)
    window = max(1, args.rolling_window)

    timesteps = list(stream_series(metrics_path, "timesteps", stride))
    mean_dist = list(stream_series(metrics_path, "mean_distance_per_step", stride))
    mean_food = list(stream_series(metrics_path, "mean_food_discovery_rate", stride))
    mean_entropy = list(stream_series(metrics_path, "mean_movement_entropy", stride))

    size = min(len(timesteps), len(mean_dist), len(mean_food), len(mean_entropy))
    if size == 0:
        print("Not enough behavioral metrics logged to plot.", file=sys.stderr)
        return 1

    timesteps_arr = np.array(timesteps[:size], dtype=float)
    mean_dist_arr = rolling_mean(mean_dist[:size], window)
    mean_food_arr = rolling_mean(mean_food[:size], window)
    mean_entropy_arr = rolling_mean(mean_entropy[:size], window)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Behavioral Metrics (down-sampled)")

    axes[0].plot(timesteps_arr, mean_dist_arr, color="#1f77b4")
    axes[0].set_ylabel("Distance/step")
    axes[0].grid(alpha=0.3)

    axes[1].plot(timesteps_arr, mean_food_arr, color="#ff7f0e")
    axes[1].set_ylabel("Food discovery rate")
    axes[1].grid(alpha=0.3)

    axes[2].plot(timesteps_arr, mean_entropy_arr, color="#2ca02c")
    axes[2].set_ylabel("Entropy (bits)")
    axes[2].set_xlabel("Timestep")
    axes[2].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = args.logs_dir / "behavioral_metrics_trends.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
