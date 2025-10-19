"""Simulation module: Main simulation loop."""

import copy
import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.agent import Agent
from src.analysis import Logger
from src.environment import Environment
from src.evolution import Evolution


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

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, config_override=resume_overrides)
        else:
            self.environment = self._create_environment()
            self.agents = self._create_initial_agents()
            self.evolution = self._create_evolution()
            self.logger = Logger(self.config["logging"])
            self._build_agent_grid()

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
                energy=agent_config["initial_energy"]
            )
            agents.append(agent)

        return agents

    def _create_evolution(self) -> Evolution:
        """Create evolution system from config."""
        evo_config = self.config["evolution"]

        return Evolution(
            mutation_rate=evo_config["mutation_rate"],
            mutation_std=evo_config["mutation_std"]
        )

    def _build_agent_grid(self) -> None:
        """Build spatial index for fast agent lookup."""
        grid_size = self.environment.grid_size
        self.agent_grid = {}

        for agent in self.agents:
            pos = agent.position
            if pos not in self.agent_grid:
                self.agent_grid[pos] = []
            self.agent_grid[pos].append(agent)

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

        # 3. Consume food
        for agent in self.agents:
            energy_gained = self.environment.consume_food(agent.position)
            agent.energy += energy_gained

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
                        offspring_pos
                    )
                    new_agents.append(offspring)

                    # Deduct energy from parent
                    agent.energy -= agent_config["reproduction_cost"]

        self.agents.extend(new_agents)

        # 6. Death and cleanup
        self.agents = [agent for agent in self.agents if agent.is_alive()]

        # 7. Population cap
        self.agents = self.evolution.select(self.agents, sim_config["population_cap"])

        # 8. Environment update
        self.environment.spawn_food()

        # 9. Logging
        if self.timestep % self.config["logging"]["log_interval"] == 0:
            self.logger.log_timestep(self.timestep, self.agents, self.environment)

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
                "position": tuple(int(v) for v in agent.position),
                "energy": float(agent.energy),
                "age": int(agent.age),
                "genome": np.array(agent.genome, copy=True),
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

        env_state = payload["environment"]
        self.environment = Environment(
            grid_size=tuple(env_state["grid_size"]),
            food_spawn_rate=env_state["food_spawn_rate"],
            food_energy_value=env_state["food_energy_value"],
            seed=None,
        )
        self.environment.grid = np.array(env_state["grid"], copy=True)

        self.agents = []
        for agent_state in payload["agents"]:
            genome = np.array(agent_state["genome"], copy=True)
            agent = Agent(
                position=tuple(agent_state["position"]),
                energy=agent_state["energy"],
                genome=genome,
            )
            agent.age = agent_state["age"]
            self.agents.append(agent)

        self.evolution = self._create_evolution()
        self.logger = Logger(self.config["logging"])

        logger_state = payload.get("logger", {})
        if logger_state:
            self.logger.metrics = copy.deepcopy(logger_state.get("metrics", self.logger.metrics))
            self.logger.prev_population = logger_state.get("prev_population", len(self.agents))

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
