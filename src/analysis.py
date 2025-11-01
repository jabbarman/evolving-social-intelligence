"""Analysis module: Metrics, logging, and analysis tools."""

import copy
import gzip
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class Logger:
    """Handles logging of simulation metrics."""

    def __init__(self, config: Dict[str, Any],
                 behavioral_config: Optional[Dict[str, Any]] = None,
                 lineage_config: Optional[Dict[str, Any]] = None):
        """Initialize logger with configuration.

        Args:
            config: Logging configuration dictionary
            behavioral_config: Behavioral metrics configuration
            lineage_config: Lineage tracking configuration
        """
        self.config = config
        self.save_dir = Path(config["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.behavioral_config = behavioral_config or {}
        self.behavioral_enabled = self.behavioral_config.get("enabled", False)
        self.behavioral_metric_keys = [
            "mean_distance_per_step",
            "std_distance_per_step",
            "median_distance_per_step",
            "mean_food_discovery_rate",
            "max_food_discovery_rate",
            "total_food_consumed",
            "mean_movement_entropy",
            "min_movement_entropy",
            "max_movement_entropy",
        ]
        self.lineage_config = lineage_config or {}

        self.metrics = {
            "timesteps": [],
            "population": [],
            "births": [],
            "deaths": [],
            "mean_energy": [],
            "mean_age": [],
            "max_age": [],
            "total_food": [],
        }

        if self.behavioral_enabled:
            for key in self.behavioral_metric_keys:
                self.metrics[key] = []

        self.prev_population = 0

    def log_timestep(self, timestep: int, agents: List, environment=None,
                     behavioral_metrics: Optional[Dict[str, float]] = None) -> None:
        """Log metrics for current timestep.

        Args:
            timestep: Current timestep
            agents: List of current agents
            environment: Environment instance (optional)
            behavioral_metrics: Optional behavioral metrics for this timestep
        """
        self.metrics["timesteps"].append(timestep)

        current_pop = len(agents)
        self.metrics["population"].append(current_pop)

        # Calculate births and deaths
        births = max(0, current_pop - self.prev_population)
        deaths = max(0, self.prev_population - current_pop)
        self.metrics["births"].append(births)
        self.metrics["deaths"].append(deaths)
        self.prev_population = current_pop

        if len(agents) > 0:
            energies = [a.energy for a in agents]
            ages = [a.age for a in agents]
            self.metrics["mean_energy"].append(np.mean(energies))
            self.metrics["mean_age"].append(np.mean(ages))
            self.metrics["max_age"].append(np.max(ages))
        else:
            self.metrics["mean_energy"].append(0)
            self.metrics["mean_age"].append(0)
            self.metrics["max_age"].append(0)

        # Count food on grid
        if environment:
            food_count = np.sum(environment.grid == 1)
            self.metrics["total_food"].append(int(food_count))
        else:
            self.metrics["total_food"].append(0)

        if self.behavioral_enabled:
            metrics = behavioral_metrics or {}
            for key in self.behavioral_metric_keys:
                value = metrics.get(key)
                if value is None:
                    self.metrics[key].append(None)
                else:
                    self.metrics[key].append(float(value))

    def save_checkpoint(self, timestep: int, simulation_state: Dict[str, Any]) -> Path:
        """Save simulation checkpoint.

        Args:
            timestep: Current timestep
            simulation_state: Complete simulation state
        """
        checkpoint_dir = self.save_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_{timestep:08d}.pkl.gz"

        payload = copy.deepcopy(simulation_state)
        payload.setdefault("timestep", timestep)
        payload.setdefault("logger", {
            "metrics": copy.deepcopy(self.metrics),
            "prev_population": self.prev_population,
        })

        with gzip.open(checkpoint_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        return checkpoint_path

    def save_metrics(self) -> None:
        """Save collected metrics to compressed NumPy format."""
        metrics_file = self.save_dir / "metrics.npz"

        arrays: Dict[str, np.ndarray] = {}
        for key, values in self.metrics.items():
            if not values:
                arrays[key] = np.array([], dtype=float)
                continue
            if any(value is None for value in values):
                arrays[key] = np.array(
                    [np.nan if value is None else value for value in values],
                    dtype=float,
                )
            else:
                arrays[key] = np.asarray(values)

        np.savez_compressed(metrics_file, **arrays)

        legacy_file = self.save_dir / "metrics.json"
        if legacy_file.exists():
            legacy_file.unlink()


def calculate_population_metrics(agents: List) -> Dict[str, float]:
    """Calculate population-level metrics.

    Args:
        agents: List of agents

    Returns:
        metrics: Dictionary of metric values
    """
    if not agents:
        return {
            "population": 0,
            "mean_energy": 0,
            "std_energy": 0,
            "mean_age": 0,
            "max_age": 0,
        }

    energies = [a.energy for a in agents]
    ages = [a.age for a in agents]

    return {
        "population": len(agents),
        "mean_energy": np.mean(energies),
        "std_energy": np.std(energies),
        "mean_age": np.mean(ages),
        "max_age": np.max(ages),
    }


def calculate_behavioral_metrics(agents: List, food_energy_value: float = 0.0) -> Dict[str, float]:
    """Calculate behavioral metrics for a snapshot of agents.

    Args:
        agents: List of agents
        food_energy_value: Energy gained per food item (optional, used for totals)

    Returns:
        Dictionary of behavioral metric values
    """
    if not agents:
        return {
            "mean_distance_per_step": 0.0,
            "std_distance_per_step": 0.0,
            "median_distance_per_step": 0.0,
            "mean_food_discovery_rate": 0.0,
            "max_food_discovery_rate": 0.0,
            "total_food_consumed": 0.0,
            "mean_movement_entropy": 0.0,
            "min_movement_entropy": 0.0,
            "max_movement_entropy": 0.0,
        }

    distances = np.array([getattr(agent, "last_move_distance", 0.0) for agent in agents], dtype=float)
    mean_distance = float(np.mean(distances)) if distances.size else 0.0
    std_distance = float(np.std(distances)) if distances.size else 0.0
    median_distance = float(np.median(distances)) if distances.size else 0.0

    discovery_rates = [getattr(agent, "discovery_rate", 0.0) for agent in agents]
    mean_food_rate = float(np.mean(discovery_rates)) if discovery_rates else 0.0
    max_food_rate = float(np.max(discovery_rates)) if discovery_rates else 0.0

    entropies = [agent.compute_movement_entropy() for agent in agents]
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0
    min_entropy = float(np.min(entropies)) if entropies else 0.0
    max_entropy = float(np.max(entropies)) if entropies else 0.0

    total_food_events = sum(getattr(agent, "food_discoveries", 0) for agent in agents)
    if food_energy_value > 0:
        total_food_consumed = float(total_food_events * food_energy_value)
    else:
        total_food_consumed = float(total_food_events)

    return {
        "mean_distance_per_step": mean_distance,
        "std_distance_per_step": std_distance,
        "median_distance_per_step": median_distance,
        "mean_food_discovery_rate": mean_food_rate,
        "max_food_discovery_rate": max_food_rate,
        "total_food_consumed": total_food_consumed,
        "mean_movement_entropy": mean_entropy,
        "min_movement_entropy": min_entropy,
        "max_movement_entropy": max_entropy,
    }


def load_metrics(save_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load metrics from the logging directory (NPZ preferred, JSON fallback)."""
    save_dir = Path(save_dir)
    metrics_npz = save_dir / "metrics.npz"
    if metrics_npz.exists():
        with np.load(metrics_npz, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    metrics_json = save_dir / "metrics.json"
    if metrics_json.exists():
        with open(metrics_json, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        return {key: np.array(values) for key, values in loaded.items()}

    raise FileNotFoundError(f"No metrics.npz or metrics.json found in {save_dir}")


def load_lineage_stats(save_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load lineage_stats.json if available."""
    stats_path = Path(save_dir) / "lineage_stats.json"
    if not stats_path.exists():
        return []
    with open(stats_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def open_lineage_database(save_dir: Union[str, Path], *, read_only: bool = True) -> sqlite3.Connection:
    """Open a connection to the lineage SQLite database."""
    db_path = Path(save_dir) / "lineage.db"
    if not db_path.exists():
        raise FileNotFoundError(f"No lineage.db found at {db_path}")
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    return sqlite3.connect(db_path)


def summarize_lineage_stats(stats: List[Dict[str, Any]]) -> Dict[str, float]:
    """Summarize lineage statistics across multiple snapshots."""
    if not stats:
        return {
            "mean_active_lineages": 0.0,
            "mean_lineage_diversity": 0.0,
            "max_generation": 0.0,
        }

    active_counts = [entry.get("active_lineages", 0) for entry in stats]
    diversity = [entry.get("lineage_diversity_index", 0.0) for entry in stats]
    max_generation = max(entry.get("max_generation", 0) for entry in stats)

    return {
        "mean_active_lineages": float(np.mean(active_counts)),
        "mean_lineage_diversity": float(np.mean(diversity)),
        "max_generation": float(max_generation),
    }
