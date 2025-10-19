"""Analysis module: Metrics, logging, and analysis tools."""

import copy
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class Logger:
    """Handles logging of simulation metrics."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize logger with configuration.

        Args:
            config: Logging configuration dictionary
        """
        self.config = config
        self.save_dir = Path(config["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

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

        self.prev_population = 0

    def log_timestep(self, timestep: int, agents: List, environment=None) -> None:
        """Log metrics for current timestep.

        Args:
            timestep: Current timestep
            agents: List of current agents
            environment: Environment instance (optional)
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
        """Save collected metrics to file."""
        metrics_file = self.save_dir / "metrics.json"

        # Convert numpy types to native Python types for JSON serialization
        metrics_clean = {}
        for key, values in self.metrics.items():
            metrics_clean[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v
                                  for v in values]

        with open(metrics_file, "w") as f:
            json.dump(metrics_clean, f, indent=2)


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


def calculate_behavioral_metrics(agents: List) -> Dict[str, float]:
    """Calculate behavioral metrics.

    Args:
        agents: List of agents

    Returns:
        metrics: Dictionary of metric values
    """
    if not agents:
        return {
            "spatial_spread": 0,
            "clustering": 0,
        }

    # Calculate spatial spread (variance in positions)
    positions = np.array([a.position for a in agents])
    spatial_spread = np.mean(np.std(positions, axis=0))

    return {
        "spatial_spread": float(spatial_spread),
        "clustering": 0,  # TODO: Implement clustering metric
    }
