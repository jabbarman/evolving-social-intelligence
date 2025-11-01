"""Simulation module: Main simulation loop."""

import copy
import gzip
import pickle
from collections import deque
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.agent import Agent
from src.analysis import Logger
from src.environment import Environment
from src.evolution import Evolution
from src.lineage import LineageTracker


class Simulation:
    """Main simulation coordinator."""

    def __init__(self, config: Dict[str, Any],
                 checkpoint_path: Optional[Union[str, Path]] = None,
                 resume_overrides: Optional[Dict[str, Any]] = None):
        """Initialize simulation with configuration.

        Args:
            config: Configuration dictionary
            checkpoint_path: Optional path to resume from checkpoint
        """
        self.config = copy.deepcopy(config)
        self.timestep = 0
        self.environment: Optional[Environment] = None
        self.agents: List[Agent] = []
        self.evolution: Optional[Evolution] = None
        self.logger: Optional[Logger] = None
        self.agent_grid: Dict[tuple, List[Agent]] = {}
        self.behavioral_config = copy.deepcopy(self.config.get("behavioral_metrics", {}))
        self.behavioral_enabled = self.behavioral_config.get("enabled", False)
        logging_cfg = self.config.get("logging", {})
        self.behavioral_log_interval = self.behavioral_config.get(
            "log_interval",
            logging_cfg.get("log_interval", 0),
        )
        self.movement_history_length = self.behavioral_config.get("movement_history_length", 20)
        self._distance_samples: List[float] = []
        self._food_consumed_this_interval: float = 0.0
        self.lineage_config = copy.deepcopy(self.config.get("lineage_tracking", {}))
        self.lineage_tracker = None

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, config_override=resume_overrides)
        else:
            self.environment = self._create_environment()
            if self.lineage_config.get("enabled", False):
                save_dir = logging_cfg.get("save_dir", "experiments/logs")
                self.lineage_tracker = LineageTracker(self.lineage_config, save_dir)
            self.agents = self._create_initial_agents()
            self.evolution = self._create_evolution()
            self.logger = Logger(
                logging_cfg,
                behavioral_config=self.behavioral_config,
                lineage_config=self.lineage_config,
            )
            self._build_agent_grid()
            if self.lineage_tracker:
                self.lineage_tracker.update_living_agents(self.agents, self.timestep)
                if self.lineage_tracker.should_log(self.timestep):
                    self.lineage_tracker.log_lineage_stats(self.timestep)

    def _create_environment(self) -> Environment:
        """Create environment from config."""
        env_config = self.config["environment"]
        sim_config = self.config["simulation"]

        return Environment(
            grid_size=tuple(sim_config["grid_size"]),
            food_spawn_rate=env_config["food_spawn_rate"],
            food_energy_value=env_config["food_energy_value"],
            seed=sim_config.get("seed")
        )

    def _create_initial_agents(self) -> List[Agent]:
        """Create initial population of agents."""
        sim_config = self.config["simulation"]
        agent_config = self.config["agent"]

        agents = []
        grid_size = sim_config["grid_size"]
        initial_pop = sim_config["initial_population"]

        for _ in range(initial_pop):
            # Random position
            position = (
                np.random.randint(0, grid_size[0]),
                np.random.randint(0, grid_size[1])
            )

            # Create agent with random brain
            agent = Agent(
                position=position,
                energy=agent_config["initial_energy"],
                movement_history_length=self.movement_history_length,
            )
            self._register_agent_lineage(agent, parent=None, birth_timestep=self.timestep)
            agents.append(agent)

        return agents

    def _create_evolution(self) -> Evolution:
        """Create evolution system from config."""
        evo_config = self.config["evolution"]

        return Evolution(
            mutation_rate=evo_config["mutation_rate"],
            mutation_std=evo_config["mutation_std"]
        )

    def _register_agent_lineage(self, agent: Agent, parent: Optional[Agent],
                                birth_timestep: int) -> None:
        """Assign lineage metadata to a newly created agent."""
        if parent is None:
            agent.parent_id = None
            agent.generation = 0
            agent.lineage_root_id = agent.id
            if self.lineage_tracker:
                self.lineage_tracker.register_founder(agent, birth_timestep)
        else:
            agent.parent_id = parent.id
            agent.generation = parent.generation + 1
            agent.lineage_root_id = parent.lineage_root_id
            parent.offspring_count += 1
            if self.lineage_tracker:
                self.lineage_tracker.register_offspring(parent, agent, birth_timestep)

    def _build_agent_grid(self) -> None:
        """Build spatial index for fast agent lookup."""
        grid_size = self.environment.grid_size
        self.agent_grid = {}

        for agent in self.agents:
            pos = agent.position
            if pos not in self.agent_grid:
                self.agent_grid[pos] = []
            self.agent_grid[pos].append(agent)

    def _should_log_behavioral(self) -> bool:
        """Check if behavioral metrics should be logged this timestep."""
        interval = self.behavioral_log_interval
        if interval is None or interval <= 0:
            interval = self.config.get("logging", {}).get("log_interval", 0)
        if interval <= 0:
            return False
        return self.timestep % interval == 0

    def _compute_behavioral_metrics(self) -> Dict[str, float]:
        """Aggregate behavioral metrics for current logging interval."""
        distances = np.array(self._distance_samples, dtype=float)
        if distances.size > 0:
            mean_distance = float(np.mean(distances))
            std_distance = float(np.std(distances))
            median_distance = float(np.median(distances))
        else:
            mean_distance = 0.0
            std_distance = 0.0
            median_distance = 0.0

        if self.agents:
            discovery_rates = [agent.discovery_rate for agent in self.agents]
            mean_discovery = float(np.mean(discovery_rates))
            max_discovery = float(np.max(discovery_rates))
            entropies = [agent.compute_movement_entropy() for agent in self.agents]
            mean_entropy = float(np.mean(entropies))
            min_entropy = float(np.min(entropies))
            max_entropy = float(np.max(entropies))
        else:
            mean_discovery = 0.0
            max_discovery = 0.0
            mean_entropy = 0.0
            min_entropy = 0.0
            max_entropy = 0.0

        return {
            "mean_distance_per_step": mean_distance,
            "std_distance_per_step": std_distance,
            "median_distance_per_step": median_distance,
            "mean_food_discovery_rate": mean_discovery,
            "max_food_discovery_rate": max_discovery,
            "total_food_consumed": float(self._food_consumed_this_interval),
            "mean_movement_entropy": mean_entropy,
            "min_movement_entropy": min_entropy,
            "max_movement_entropy": max_entropy,
        }

    def _reset_behavioral_accumulators(self) -> None:
        """Reset accumulated behavioral statistics after logging."""
        self._distance_samples.clear()
        self._food_consumed_this_interval = 0.0

    def step(self) -> None:
        """Execute one simulation timestep."""
        agent_config = self.config["agent"]
        sim_config = self.config["simulation"]

        # Rebuild spatial index
        self._build_agent_grid()

        # 1. Agent perception and decision
        actions = []
        for agent in self.agents:
            observations = self._perceive_fast(agent)
            action_output = agent.decide(observations)
            actions.append(action_output)

        # 2. Execute movement actions
        for agent, action_output in zip(self.agents, actions):
            # Use softmax to select movement action (first 5 outputs)
            movement_logits = action_output[:5]
            movement_probs = np.exp(movement_logits) / np.sum(np.exp(movement_logits))
            movement_action = np.random.choice(5, p=movement_probs)

            # Execute movement
            agent.move(movement_action, self.environment.grid_size)

            # Energy cost for movement
            if movement_action != 4:  # If not staying
                agent.update_energy(agent_config["movement_cost"])

        if self.behavioral_enabled:
            self._distance_samples.extend(agent.last_move_distance for agent in self.agents)

        # 3. Consume food
        for agent in self.agents:
            energy_gained = self.environment.consume_food(agent.position)
            agent.energy += energy_gained
            if energy_gained > 0:
                agent.record_food_discovery()
                if self.behavioral_enabled:
                    self._food_consumed_this_interval += energy_gained

        # 4. Update energy (base metabolic cost)
        for agent in self.agents:
            agent.update_energy(agent_config["base_metabolic_cost"])

        # 5. Reproduction
        new_agents = []
        for agent in self.agents:
            if agent.energy > agent_config["reproduction_threshold"]:
                # Find adjacent empty cell
                offspring_pos = self._find_adjacent_position(agent.position)
                if offspring_pos is not None:
                    # Create offspring
                    offspring = self.evolution.reproduce(
                        agent,
                        agent_config["reproduction_cost"],
                        offspring_pos,
                        movement_history_length=self.movement_history_length,
                    )
                    self._register_agent_lineage(offspring, parent=agent, birth_timestep=self.timestep)
                    new_agents.append(offspring)

                    # Deduct energy from parent
                    agent.energy -= agent_config["reproduction_cost"]

        self.agents.extend(new_agents)

        # 6. Death and cleanup
        self.agents = [agent for agent in self.agents if agent.is_alive()]

        # 7. Population cap
        self.agents = self.evolution.select(self.agents, sim_config["population_cap"])

        if self.lineage_tracker:
            self.lineage_tracker.update_living_agents(self.agents, self.timestep)
            if self.lineage_tracker.should_log(self.timestep):
                self.lineage_tracker.log_lineage_stats(self.timestep)

        # 8. Environment update
        self.environment.spawn_food()

        # 9. Logging
        logging_interval = self.config.get("logging", {}).get("log_interval", 0)
        if logging_interval and self.timestep % logging_interval == 0:
            behavioral_metrics = None
            if self.behavioral_enabled and self._should_log_behavioral():
                behavioral_metrics = self._compute_behavioral_metrics()
                self._reset_behavioral_accumulators()
            self.logger.log_timestep(
                self.timestep,
                self.agents,
                self.environment,
                behavioral_metrics=behavioral_metrics,
            )

    def _perceive_fast(self, agent) -> np.ndarray:
        """Optimized perception using spatial indexing.

        Returns:
            observations: 52-dimensional vector (25 food + 25 agents + energy + age)
        """
        x, y = agent.position
        perception_range = 2  # 2 cells in each direction = 5x5 grid
        grid_size = self.environment.grid_size

        # Initialize observation grids
        food_obs = np.zeros(25)
        agent_obs = np.zeros(25)

        # Scan 5x5 local area
        idx = 0
        for dx in range(-perception_range, perception_range + 1):
            for dy in range(-perception_range, perception_range + 1):
                # Get position with toroidal wrapping
                pos_x = (x + dx) % grid_size[0]
                pos_y = (y + dy) % grid_size[1]
                pos = (pos_x, pos_y)

                # Check for food
                if self.environment.grid[pos_x, pos_y] == 1:
                    food_obs[idx] = 1.0

                # Check for other agents using spatial index
                if pos in self.agent_grid:
                    for other_agent in self.agent_grid[pos]:
                        if other_agent is not agent:
                            agent_obs[idx] = 1.0
                            break

                idx += 1

        # Normalize energy and age for better neural network performance
        normalized_energy = agent.energy / 100.0
        normalized_age = min(agent.age / 1000.0, 1.0)

        # Concatenate all observations
        observations = np.concatenate([
            food_obs,
            agent_obs,
            [normalized_energy],
            [normalized_age]
        ])

        return observations

    def _find_adjacent_position(self, position: tuple) -> tuple:
        """Find an empty adjacent position for offspring."""
        x, y = position
        grid_size = self.environment.grid_size

        # Check adjacent cells
        adjacent_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        np.random.shuffle(adjacent_offsets)

        for dx, dy in adjacent_offsets:
            new_x = (x + dx) % grid_size[0]
            new_y = (y + dy) % grid_size[1]
            new_pos = (new_x, new_y)

            # Check if position is empty (no agents there)
            occupied = any(agent.position == new_pos for agent in self.agents)
            if not occupied:
                return new_pos

        return None  # No empty adjacent cell

    def run(self, num_steps: int = None) -> None:
        """Run simulation for specified number of steps.

        Args:
            num_steps: Number of timesteps to run (None = use config)
        """
        if num_steps is None:
            num_steps = self.config["simulation"]["max_timesteps"]

        for _ in range(num_steps):
            self.step()
            self.timestep += 1
            self._maybe_save_checkpoint()

    def save_checkpoint(self, checkpoint_path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the full simulation state to disk.

        Args:
            checkpoint_path: Optional explicit path. Defaults to logging directory.

        Returns:
            Path to the written checkpoint file.
        """
        payload = self._build_checkpoint_payload()

        if checkpoint_path is None:
            if not self.logger:
                raise RuntimeError("Logger must be initialized before saving checkpoints.")
            return self.logger.save_checkpoint(self.timestep, payload)

        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(checkpoint_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Union[str, Path],
                        config_override: Optional[Dict[str, Any]] = None) -> None:
        """Restore simulation state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            config_override: Optional overrides applied on top of stored config.
        """
        payload = self._load_checkpoint_payload(checkpoint_path)
        self._restore_from_payload(payload, config_override=config_override)

    def _maybe_save_checkpoint(self) -> Optional[Path]:
        """Save checkpoint if configured interval is reached."""
        logging_cfg = self.config.get("logging", {})
        interval = logging_cfg.get("checkpoint_interval")
        if not interval or interval <= 0:
            return None
        if self.timestep <= 0 or self.timestep % interval != 0:
            return None
        return self.save_checkpoint()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path],
                        config_override: Optional[Dict[str, Any]] = None) -> "Simulation":
        """Create simulation instance directly from checkpoint."""
        payload = cls._load_checkpoint_payload(checkpoint_path)
        config = copy.deepcopy(payload["config"])
        if config_override:
            config = cls._deep_update(config, config_override)
        sim = cls(config)
        sim._restore_from_payload(payload, config_override=config_override)
        return sim

    def _build_checkpoint_payload(self) -> Dict[str, Any]:
        """Create a serializable snapshot of current state."""
        if not self.environment or not self.evolution or not self.logger:
            raise RuntimeError("Simulation components not initialized; cannot checkpoint.")

        environment_state = {
            "grid_size": tuple(self.environment.grid_size),
            "food_spawn_rate": self.environment.food_spawn_rate,
            "food_energy_value": self.environment.food_energy_value,
            "grid": np.array(self.environment.grid, copy=True),
        }

        agents_state = []
        for agent in self.agents:
            agents_state.append({
                "id": int(agent.id),
                "position": tuple(int(v) for v in agent.position),
                "energy": float(agent.energy),
                "age": int(agent.age),
                "genome": np.array(agent.genome, copy=True),
                "parent_id": agent.parent_id,
                "generation": int(agent.generation),
                "lineage_root_id": int(agent.lineage_root_id),
                "offspring_count": int(agent.offspring_count),
                "movement_history_length": int(agent.movement_history_length),
                "total_moves": int(agent.total_moves),
                "food_discoveries": int(agent.food_discoveries),
                "discovery_rate": float(agent.discovery_rate),
                "recent_actions": list(agent.recent_actions),
                "movement_entropy": float(agent.movement_entropy),
                "last_move_distance": float(agent.last_move_distance),
            })

        logger_state = {
            "metrics": copy.deepcopy(self.logger.metrics),
            "prev_population": int(self.logger.prev_population),
        }

        payload: Dict[str, Any] = {
            "version": 1,
            "config": copy.deepcopy(self.config),
            "timestep": int(self.timestep),
            "environment": environment_state,
            "agents": agents_state,
            "rng_state": np.random.get_state(),
            "logger": logger_state,
            "lineage": self.lineage_tracker.to_state() if self.lineage_tracker else None,
        }
        return payload

    @staticmethod
    def _load_checkpoint_payload(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoint payload from disk."""
        checkpoint_path = Path(checkpoint_path)
        with gzip.open(checkpoint_path, "rb") as fh:
            payload = pickle.load(fh)
        return payload

    def _restore_from_payload(self, payload: Dict[str, Any],
                              config_override: Optional[Dict[str, Any]] = None) -> None:
        """Restore instance attributes from checkpoint payload."""
        config = copy.deepcopy(payload["config"])
        if config_override:
            config = self._deep_update(config, config_override)
        self.config = config
        self.behavioral_config = copy.deepcopy(self.config.get("behavioral_metrics", {}))
        self.behavioral_enabled = self.behavioral_config.get("enabled", False)
        logging_cfg = self.config.get("logging", {})
        self.behavioral_log_interval = self.behavioral_config.get(
            "log_interval",
            logging_cfg.get("log_interval", 0),
        )
        self.movement_history_length = self.behavioral_config.get("movement_history_length", 20)
        self.lineage_config = copy.deepcopy(self.config.get("lineage_tracking", {}))
        save_dir = logging_cfg.get("save_dir", "experiments/logs")
        self.lineage_tracker = LineageTracker.from_state(
            self.lineage_config,
            save_dir,
            payload.get("lineage"),
        )
        self._distance_samples = []
        self._food_consumed_this_interval = 0.0

        env_state = payload["environment"]
        self.environment = Environment(
            grid_size=tuple(env_state["grid_size"]),
            food_spawn_rate=env_state["food_spawn_rate"],
            food_energy_value=env_state["food_energy_value"],
            seed=None,
        )
        self.environment.grid = np.array(env_state["grid"], copy=True)

        self.agents = []
        max_agent_id = -1
        for agent_state in payload["agents"]:
            genome = np.array(agent_state["genome"], copy=True)
            movement_history_length = agent_state.get("movement_history_length", self.movement_history_length)
            agent = Agent(
                position=tuple(agent_state["position"]),
                energy=agent_state["energy"],
                genome=genome,
                movement_history_length=movement_history_length,
                agent_id=agent_state.get("id"),
                parent_id=agent_state.get("parent_id"),
                generation=agent_state.get("generation", 0),
            )
            agent.age = agent_state["age"]
            agent.offspring_count = agent_state.get("offspring_count", 0)
            agent.lineage_root_id = agent_state.get("lineage_root_id", agent.id)
            agent.total_moves = agent_state.get("total_moves", 0)
            agent.food_discoveries = agent_state.get("food_discoveries", 0)
            agent.discovery_rate = agent_state.get("discovery_rate", 0.0)
            agent.movement_entropy = agent_state.get("movement_entropy", 0.0)
            agent.last_move_distance = agent_state.get("last_move_distance", 0.0)
            recent_actions = agent_state.get("recent_actions", [])
            agent.recent_actions = deque(recent_actions, maxlen=agent.movement_history_length)
            self.agents.append(agent)
            max_agent_id = max(max_agent_id, agent.id)

        self.evolution = self._create_evolution()
        self.logger = Logger(
            logging_cfg,
            behavioral_config=self.behavioral_config,
            lineage_config=self.lineage_config,
        )

        logger_state = payload.get("logger", {})
        if logger_state:
            self.logger.metrics = copy.deepcopy(logger_state.get("metrics", self.logger.metrics))
            self.logger.prev_population = logger_state.get("prev_population", len(self.agents))

        next_id = max_agent_id + 1
        if self.lineage_tracker and self.lineage_tracker.enabled:
            next_id = max(next_id, self.lineage_tracker.next_id)
            self.lineage_tracker.update_living_agents(self.agents, int(payload["timestep"]))
        Agent._id_counter = count(next_id)

        self.timestep = int(payload["timestep"])
        np.random.set_state(payload["rng_state"])
        self._build_agent_grid()

    @staticmethod
    def _deep_update(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries."""
        for key, value in updates.items():
            if (key in original and isinstance(original[key], dict)
                    and isinstance(value, dict)):
                original[key] = Simulation._deep_update(original[key], value)
            else:
                original[key] = value
        return original
