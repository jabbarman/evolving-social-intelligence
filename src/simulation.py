"""Simulation module: Main simulation loop."""

import numpy as np
from typing import Dict, Any, List
from src.environment import Environment
from src.agent import Agent
from src.evolution import Evolution
from src.analysis import Logger


class Simulation:
    """Main simulation coordinator."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize simulation with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.timestep = 0

        # Initialize components
        self.environment = self._create_environment()
        self.agents = self._create_initial_agents()
        self.evolution = self._create_evolution()
        self.logger = Logger(config["logging"])

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

    def step(self) -> None:
        """Execute one simulation timestep."""
        agent_config = self.config["agent"]
        sim_config = self.config["simulation"]

        # 1. Agent perception and decision
        actions = []
        for agent in self.agents:
            observations = agent.perceive(self.environment, self.agents)
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
            self.logger.log_timestep(self.timestep, self.agents)

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
