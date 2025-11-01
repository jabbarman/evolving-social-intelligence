#!/usr/bin/env python3
"""Down-sampled behavioral metrics plots for large runs.

Usage:
    python scripts/plot_behavioral_trends.py [--logs-dir experiments/logs]

The script works with the new NumPy-based metrics storage (metrics.npz) and
falls back to the legacy metrics.json if required. Down-sampling and rolling
averages keep the plots readable even for long simulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # ensure headless rendering before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("experiments/logs"),
        help="Directory containing metrics.npz (default: experiments/logs)",
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


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with bounded window."""
    if values.size == 0:
        return values
    window = max(1, min(window, values.size))
    if window == 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    stride = max(1, args.stride)
    window = max(1, args.rolling_window)
    logs_dir = args.logs_dir

    plots_dir = logs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_npz = logs_dir / "metrics.npz"
    metrics_json = logs_dir / "metrics.json"

    if metrics_npz.exists():
        with np.load(metrics_npz, allow_pickle=False, mmap_mode="r") as data:
            raw_timesteps = np.asarray(data["timesteps"], dtype=float)
            length = raw_timesteps.size

            def _arr(key: str) -> np.ndarray:
                if key not in data.files:
                    return np.full(length, np.nan, dtype=float)
                arr = np.asarray(data[key], dtype=float)
                if arr.size != length:
                    raise SystemExit(f"metrics field '{key}' length {arr.size} mismatches timesteps {length}")
                return arr

            raw_mean_dist = _arr("mean_distance_per_step")
            raw_mean_food = _arr("mean_food_discovery_rate")
            raw_mean_entropy = _arr("mean_movement_entropy")
    elif metrics_json.exists():
        import json

        loaded = json.loads(metrics_json.read_text())
        raw_timesteps = np.asarray(loaded.get("timesteps", []), dtype=float)
        length = raw_timesteps.size

        def _json_arr(key: str) -> np.ndarray:
            values = loaded.get(key)
            if values is None:
                return np.full(length, np.nan, dtype=float)
            arr = np.asarray([np.nan if value is None else value for value in values], dtype=float)
            if arr.size != length:
                raise SystemExit(f"metrics field '{key}' length {arr.size} mismatches timesteps {length}")
            return arr

        raw_mean_dist = _json_arr("mean_distance_per_step")
        raw_mean_food = _json_arr("mean_food_discovery_rate")
        raw_mean_entropy = _json_arr("mean_movement_entropy")
    else:
        raise SystemExit(f"No metrics.npz found at {metrics_npz}")

    if raw_timesteps.size == 0:
        raise SystemExit("Not enough behavioral metrics logged to plot.")

    timesteps = raw_timesteps[::stride]
    mean_dist = raw_mean_dist[::stride]
    mean_food = raw_mean_food[::stride]
    mean_entropy = raw_mean_entropy[::stride]

    mask = np.isfinite(mean_dist) & np.isfinite(mean_food) & np.isfinite(mean_entropy)
    timesteps = timesteps[mask]
    mean_dist = mean_dist[mask]
    mean_food = mean_food[mask]
    mean_entropy = mean_entropy[mask]

    if timesteps.size == 0:
        raise SystemExit("Metrics contain no finite values to plot.")

    effective_window = max(1, min(window, mean_dist.size, mean_food.size, mean_entropy.size))

    mean_dist_arr = rolling_mean(mean_dist.astype(float), effective_window)
    mean_food_arr = rolling_mean(mean_food.astype(float), effective_window)
    mean_entropy_arr = rolling_mean(mean_entropy.astype(float), effective_window)

    valid_len = min(mean_dist_arr.size, mean_food_arr.size, mean_entropy_arr.size)
    offset = max(0, effective_window - 1)
    timesteps_arr = timesteps[offset:offset + valid_len]
    mean_dist_arr = mean_dist_arr[:valid_len]
    mean_food_arr = mean_food_arr[:valid_len]
    mean_entropy_arr = mean_entropy_arr[:valid_len]

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
    output_path = plots_dir / "behavioral_metrics_trends.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
