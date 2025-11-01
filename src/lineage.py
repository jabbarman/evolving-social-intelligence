"""Lineage tracking utilities backed by SQLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union

import numpy as np

from src.lineage_db import LineageDatabase, ensure_database

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from src.agent import Agent  # pylint: disable=cyclic-import


class LineageTracker:
    """Maintain agent lineage information and derived statistics."""

    def __init__(self, config: Dict[str, Any], save_dir: Union[str, Path],
                 *, reset_db: bool = False):
        """Initialize tracker with configuration and log directory."""
        self.config = dict(config or {})
        self.enabled = self.config.get("enabled", False)
        self.save_interval = int(self.config.get("save_interval", 0))
        self.track_genetic_distance = bool(self.config.get("track_genetic_distance", False))
        self.max_lineage_depth = int(self.config.get("max_lineage_depth", 1000))

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.lineage_stats_path = self.save_dir / "lineage_stats.json"
        self.db_path = self.save_dir / "lineage.db"
        self.db: Optional[LineageDatabase] = None
        if self.enabled:
            self.db = ensure_database(self.db_path, reset=reset_db)

        self.stats_history: List[Dict[str, Any]] = []
        self.last_saved_timestep: Optional[int] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_founder(self, agent: "Agent", timestep: int) -> None:
        """Register a founding agent (no parent)."""
        if not self.enabled or not self.db:
            return
        agent.parent_id = None
        agent.generation = 0
        agent.lineage_root_id = agent.id
        self.db.register_founder(agent.id, timestep)

    def register_offspring(self, parent: "Agent", offspring: "Agent", timestep: int) -> None:
        """Register offspring and update lineage relationships."""
        if not self.enabled or not self.db:
            return
        offspring.parent_id = parent.id
        offspring.generation = parent.generation + 1
        offspring.lineage_root_id = parent.lineage_root_id
        self.db.register_offspring(
            agent_id=offspring.id,
            parent_id=parent.id,
            root_id=offspring.lineage_root_id,
            generation=offspring.generation,
            birth=timestep,
        )

    # ------------------------------------------------------------------
    # Lifecycle updates
    # ------------------------------------------------------------------
    def update_living_agents(self, agents: Iterable["Agent"], timestep: int) -> None:
        """Update living agent registry and note deaths."""
        if not self.enabled or not self.db:
            return
        current_ids = {agent.id for agent in agents}
        living_ids = set(self.db.living_ids())
        dead_ids = [agent_id for agent_id in living_ids if agent_id not in current_ids]
        self.db.mark_deaths(dead_ids, timestep)

        generation_updates = [(agent.id, agent.generation) for agent in agents]
        self.db.update_generations(generation_updates)

    def max_agent_id(self) -> int:
        if not self.enabled or not self.db:
            return -1
        return self.db.max_agent_id()

    # ------------------------------------------------------------------
    # Logging / persistence
    # ------------------------------------------------------------------
    def should_log(self, timestep: int) -> bool:
        """Determine if lineage metrics should be logged at this timestep."""
        if not self.enabled or self.save_interval <= 0:
            return False
        return timestep % self.save_interval == 0

    def log_lineage_stats(self, timestep: int) -> Optional[Dict[str, Any]]:
        """Compute and persist lineage statistics."""
        if not self.enabled or not self.db:
            return None

        stats = self._build_lineage_stats(timestep)
        self.stats_history.append(stats)
        self.last_saved_timestep = timestep
        self._write_stats_file()
        return stats

    def _build_lineage_stats(self, timestep: int) -> Dict[str, Any]:
        """Build lineage statistics payload for current timestep."""
        if not self.db:
            return {}

        root_counts = self.db.root_counts()
        founders = self.db.founders()
        descendants_map = {founder_id: root_counts.get(founder_id, 0) for founder_id in founders}
        total_population = int(sum(root_counts.values()))

        active_lineages = sum(1 for count in descendants_map.values() if count > 0)
        extinct_lineages = len(descendants_map) - active_lineages
        dominant_lineages = sorted(
            [
                {
                    "founder_id": founder_id,
                    "descendants": count,
                    "percentage": (float(count) / total_population * 100.0) if total_population else 0.0,
                }
                for founder_id, count in descendants_map.items()
            ],
            key=lambda entry: entry["descendants"],
            reverse=True,
        )[:5]

        generations = self.db.living_generations()
        mean_generation = float(np.mean(generations)) if generations else 0.0
        max_generation = int(np.max(generations)) if generations else 0

        diversity_index = self._compute_diversity_index(descendants_map, total_population)

        return {
            "timestep": int(timestep),
            "total_agents": int(total_population),
            "active_lineages": int(active_lineages),
            "extinct_lineages": int(extinct_lineages),
            "dominant_lineages": dominant_lineages,
            "lineage_diversity_index": diversity_index,
            "mean_generation": mean_generation,
            "max_generation": max_generation,
            "descendants_per_founder": descendants_map,
        }

    @staticmethod
    def _compute_diversity_index(descendants_map: Dict[int, int], total_population: int) -> float:
        """Compute Simpson's diversity index for lineages."""
        if total_population <= 0:
            return 0.0
        proportions = [
            count / total_population
            for count in descendants_map.values()
            if count > 0
        ]
        if not proportions:
            return 0.0
        index = 1.0 - sum(p ** 2 for p in proportions)
        return float(index)

    def _write_stats_file(self) -> None:
        """Persist lineage statistics history to disk."""
        if not self.enabled:
            return
        serializable = []
        for entry in self.stats_history:
            serializable.append(self._serialize_entry(entry))
        with open(self.lineage_stats_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)

    @staticmethod
    def _serialize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to native Python types for JSON serialization."""
        serialized = {}
        for key, value in entry.items():
            if isinstance(value, np.generic):
                serialized[key] = value.item()
            elif isinstance(value, list):
                serialized[key] = [
                    LineageTracker._serialize_entry(item) if isinstance(item, dict) else item
                    for item in value
                ]
            elif isinstance(value, dict):
                serialized[key] = {
                    key_inner: LineageTracker._serialize_entry(value_inner) if isinstance(value_inner, dict)
                    else value_inner
                    for key_inner, value_inner in value.items()
                }
            else:
                serialized[key] = value
        return serialized

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        """Serialize lineage tracker state for checkpointing."""
        if not self.enabled:
            return {}
        return {
            "config": self.config,
            "stats_history": self.stats_history,
            "last_saved_timestep": self.last_saved_timestep,
            "db_path": str(self.db_path),
        }

    @classmethod
    def from_state(cls, config: Dict[str, Any], save_dir: Union[str, Path],
                   state: Optional[Dict[str, Any]]) -> "LineageTracker":
        """Restore tracker from serialized state."""
        tracker = cls(config, save_dir, reset_db=False)
        if not state:
            return tracker

        tracker.stats_history = state.get("stats_history", [])
        tracker.last_saved_timestep = state.get("last_saved_timestep")
        db_path_str = state.get("db_path")
        if tracker.enabled and db_path_str:
            tracker.db_path = Path(db_path_str)
            tracker.db = ensure_database(tracker.db_path, reset=False)
        return tracker

    def close(self) -> None:
        if self.db:
            self.db.close()
