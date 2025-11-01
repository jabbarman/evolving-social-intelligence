"""Lineage tracking utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from src.agent import Agent  # pylint: disable=cyclic-import


class LineageTracker:
    """Maintain agent lineage information and derived statistics."""

    def __init__(self, config: Dict[str, Any], save_dir: Union[str, Path]):
        """Initialize tracker with configuration and log directory."""
        self.config = dict(config or {})
        self.enabled = self.config.get("enabled", False)
        self.save_interval = int(self.config.get("save_interval", 0))
        self.track_genetic_distance = bool(self.config.get("track_genetic_distance", False))
        self.max_lineage_depth = int(self.config.get("max_lineage_depth", 1000))

        self.save_dir = Path(save_dir)
        self.lineage_tree_path = self.save_dir / "lineage_tree.json"
        self.lineage_stats_path = self.save_dir / "lineage_stats.json"

        # Internal state
        self.agents: Dict[int, Dict[str, Any]] = {}
        self.founding_ids: List[int] = []
        self.living_ids: Set[int] = set()
        self.stats_history: List[Dict[str, Any]] = []
        self.last_saved_timestep: Optional[int] = None
        self.next_id: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_founder(self, agent: "Agent", timestep: int) -> None:
        """Register a founding agent (no parent)."""
        if not self.enabled:
            return
        agent.parent_id = None
        agent.generation = 0
        agent.lineage_root_id = agent.id
        self._register_agent(agent, parent_id=None, timestep=timestep)
        self.founding_ids.append(agent.id)

    def register_offspring(self, parent: "Agent", offspring: "Agent", timestep: int) -> None:
        """Register offspring and update lineage relationships."""
        if not self.enabled:
            return
        offspring.parent_id = parent.id
        offspring.generation = parent.generation + 1
        offspring.lineage_root_id = parent.lineage_root_id
        self._register_agent(offspring, parent_id=parent.id, timestep=timestep)

    def _register_agent(self, agent: "Agent", parent_id: Optional[int], timestep: int) -> None:
        """Internal helper to register agent metadata."""
        if not self.enabled:
            return
        parent_record = self.agents.get(parent_id) if parent_id is not None else None
        root_id = parent_record["root"] if parent_record else agent.id

        self.agents[agent.id] = {
            "parent": parent_id,
            "generation": agent.generation,
            "root": root_id,
            "children": set(),
            "birth": timestep,
            "death": None,
        }

        if parent_record is not None:
            parent_record["children"].add(agent.id)

        self.living_ids.add(agent.id)
        self._update_next_id(agent.id)

    def _update_next_id(self, agent_id: int) -> None:
        """Ensure the next available ID is always ahead of current assignments."""
        self.next_id = max(self.next_id, agent_id + 1)

    # ------------------------------------------------------------------
    # Lifecycle updates
    # ------------------------------------------------------------------
    def update_living_agents(self, agents: Iterable["Agent"], timestep: int) -> None:
        """Update living agent registry and note deaths."""
        if not self.enabled:
            return
        current_ids = {agent.id for agent in agents}

        # Record deaths
        for dead_id in self.living_ids - current_ids:
            record = self.agents.get(dead_id)
            if record and record["death"] is None:
                record["death"] = timestep

        # Ensure metadata stays current
        for agent in agents:
            record = self.agents.get(agent.id)
            if record:
                record["generation"] = agent.generation

        self.living_ids = current_ids

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
        if not self.enabled:
            return None

        stats = self._build_lineage_stats(timestep)
        self.stats_history.append(stats)
        self.last_saved_timestep = timestep
        self._write_stats_file()
        self._write_tree_file()
        return stats

    def _build_lineage_stats(self, timestep: int) -> Dict[str, Any]:
        """Build lineage statistics payload for current timestep."""
        total_population = len(self.living_ids)
        descendants_map = {founder_id: 0 for founder_id in self.founding_ids}

        for agent_id in self.living_ids:
            record = self.agents.get(agent_id)
            if not record:
                continue
            root_id = record["root"]
            descendants_map[root_id] = descendants_map.get(root_id, 0) + 1

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

        generations = [
            self.agents[agent_id]["generation"]
            for agent_id in self.living_ids
            if agent_id in self.agents
        ]
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
        self.save_dir.mkdir(parents=True, exist_ok=True)
        serializable = []
        for entry in self.stats_history:
            serializable.append(self._serialize_entry(entry))
        with open(self.lineage_stats_path, "w", encoding="utf-8") as fh:
            json.dump(serializable, fh, indent=2)

    def _write_tree_file(self) -> None:
        """Persist full lineage tree to disk."""
        if not self.enabled:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        agents_payload = {}
        for agent_id, record in self.agents.items():
            agents_payload[str(agent_id)] = {
                "parent": record["parent"],
                "generation": record["generation"],
                "root": record["root"],
                "offspring": sorted(record["children"]),
                "birth": record["birth"],
                "death": record["death"],
            }

        payload = {
            "agents": agents_payload,
            "founding_population": list(self.founding_ids),
            "current_population": sorted(self.living_ids),
        }
        with open(self.lineage_tree_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

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
        agents_state = {}
        for agent_id, record in self.agents.items():
            agents_state[str(agent_id)] = {
                "parent": record["parent"],
                "generation": record["generation"],
                "root": record["root"],
                "children": list(record["children"]),
                "birth": record["birth"],
                "death": record["death"],
            }

        return {
            "config": self.config,
            "agents": agents_state,
            "founding_ids": list(self.founding_ids),
            "living_ids": list(self.living_ids),
            "stats_history": self.stats_history,
            "last_saved_timestep": self.last_saved_timestep,
            "next_id": self.next_id,
        }

    @classmethod
    def from_state(cls, config: Dict[str, Any], save_dir: Union[str, Path],
                   state: Optional[Dict[str, Any]]) -> "LineageTracker":
        """Restore tracker from serialized state."""
        tracker = cls(config, save_dir)
        if not state:
            return tracker

        tracker.agents = {}
        agents_state = state.get("agents", {})
        for agent_id_str, record in agents_state.items():
            agent_id = int(agent_id_str)
            tracker.agents[agent_id] = {
                "parent": record.get("parent"),
                "generation": record.get("generation", 0),
                "root": record.get("root", agent_id),
                "children": set(record.get("children", [])),
                "birth": record.get("birth"),
                "death": record.get("death"),
            }
        tracker.founding_ids = [int(agent_id) for agent_id in state.get("founding_ids", [])]
        tracker.living_ids = {int(agent_id) for agent_id in state.get("living_ids", [])}
        tracker.stats_history = state.get("stats_history", [])
        tracker.last_saved_timestep = state.get("last_saved_timestep")
        tracker.next_id = int(state.get("next_id", 0))
        max_existing = max(tracker.agents.keys(), default=-1)
        tracker.next_id = max(tracker.next_id, max_existing + 1)
        return tracker
