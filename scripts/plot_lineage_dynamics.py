#!/usr/bin/env python3
"""Visualize lineage dynamics from lineage_stats.json.

Usage:
    python scripts/plot_lineage_dynamics.py [--logs-dir experiments/logs]

The script produces two figures:
  * lineage_metrics_summary.png – active/extinct lineages, Simpson diversity,
    and mean generation depth over time.
  * dominant_lineages_latest.png – bar chart of the most dominant founders in
    the most recent snapshot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering before importing pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("experiments/logs"),
        help="Directory containing lineage_stats.json (default: experiments/logs)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    plots_dir = args.logs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.logs_dir / "lineage_stats.json"

    if not stats_path.exists():
        print(f"No lineage_stats.json found at {stats_path}", file=sys.stderr)
        return 1

    lineage_stats = json.loads(stats_path.read_text())
    if not lineage_stats:
        print("lineage_stats.json is empty.", file=sys.stderr)
        return 1

    timesteps = np.array([entry.get("timestep", 0) for entry in lineage_stats], dtype=float)
    active = np.array([entry.get("active_lineages", 0) for entry in lineage_stats], dtype=float)
    extinct = np.array([entry.get("extinct_lineages", 0) for entry in lineage_stats], dtype=float)
    diversity = np.array([entry.get("lineage_diversity_index", 0.0) for entry in lineage_stats], dtype=float)
    mean_generation = np.array([entry.get("mean_generation", 0.0) for entry in lineage_stats], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Lineage Dynamics")

    axes[0].plot(timesteps, active, marker="o", color="#9467bd", label="Active")
    axes[0].plot(timesteps, extinct, marker="o", color="#8c564b", label="Extinct")
    axes[0].set_ylabel("Lineage count")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(timesteps, diversity, marker="o", color="#17becf")
    axes[1].set_ylabel("Simpson diversity")
    axes[1].grid(alpha=0.3)

    axes[2].plot(timesteps, mean_generation, marker="o", color="#d62728")
    axes[2].set_ylabel("Mean generation")
    axes[2].set_xlabel("Timestep")
    axes[2].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    summary_path = plots_dir / "lineage_metrics_summary.png"
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {summary_path}")

    latest = lineage_stats[-1]
    dominant = latest.get("dominant_lineages", [])
    if dominant:
        founders = [str(entry.get("founder_id", "")) for entry in dominant]
        descendants = [entry.get("descendants", 0) for entry in dominant]
        percentages = [entry.get("percentage", 0.0) for entry in dominant]

        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(founders, descendants, color="#1f77b4")
        ax.set_title(f"Dominant Lineages at Timestep {latest.get('timestep', 0)}")
        ax.set_xlabel("Founder ID")
        ax.set_ylabel("Living descendants")
        ax.grid(axis="y", alpha=0.3)

        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        dominant_path = plots_dir / "dominant_lineages_latest.png"
        fig.savefig(dominant_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {dominant_path}")
    else:
        print("No dominant lineage data found in latest snapshot.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
