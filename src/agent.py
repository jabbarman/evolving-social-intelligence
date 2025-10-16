"""Agent module: Agent class with sensors and actuators."""

import numpy as np
from typing import Tuple, Optional
from src.brain import Brain


class Agent:
    """An agent that can perceive, decide, and act in the environment."""

    def __init__(self, position: Tuple[int, int], energy: float,
                 genome: Optional[np.ndarray] = None, brain: Optional[Brain] = None):
        """Initialize an agent.

        Args:
            position: (x, y) position in the grid
            energy: Initial energy level
            genome: Neural network weights (optional, random if not provided)
            brain: Brain instance (optional, created if not provided)
        """
        self.position = position
        self.energy = energy
        self.age = 0

        # Create brain if not provided
        if brain is None:
            self.brain = Brain()
        else:
            self.brain = brain

        # Set genome
        if genome is not None:
            self.brain.set_weights(genome)

        self.genome = self.brain.get_weights()

    def perceive(self, environment, agents) -> np.ndarray:
        """Get local observations (5x5 grid).

        Returns:
            observations: 52-dimensional vector (25 food + 25 agents + energy + age)
        """
        x, y = self.position
        perception_range = 2  # 2 cells in each direction = 5x5 grid

        # Initialize observation grids
        food_obs = np.zeros(25)
        agent_obs = np.zeros(25)

        # Scan 5x5 local area
        idx = 0
        for dx in range(-perception_range, perception_range + 1):
            for dy in range(-perception_range, perception_range + 1):
                # Get position with toroidal wrapping
                pos_x = (x + dx) % environment.grid_size[0]
                pos_y = (y + dy) % environment.grid_size[1]

                # Check for food
                if environment.grid[pos_x, pos_y] == 1:
                    food_obs[idx] = 1.0

                # Check for other agents
                for agent in agents:
                    if agent is not self and agent.position == (pos_x, pos_y):
                        agent_obs[idx] = 1.0
                        break

                idx += 1

        # Normalize energy and age for better neural network performance
        normalized_energy = self.energy / 100.0
        normalized_age = min(self.age / 1000.0, 1.0)

        # Concatenate all observations
        observations = np.concatenate([
            food_obs,
            agent_obs,
            [normalized_energy],
            [normalized_age]
        ])

        return observations

    def decide(self, observations: np.ndarray) -> np.ndarray:
        """Process observations through neural network.

        Args:
            observations: 52-dimensional input vector

        Returns:
            actions: 6-dimensional output (5 movement + 1 communication)
        """
        return self.brain.forward(observations)

    def move(self, action: int, grid_size: Tuple[int, int]) -> None:
        """Move the agent based on action (0=up, 1=down, 2=left, 3=right, 4=stay).

        Args:
            action: Movement action index
            grid_size: Size of the grid for toroidal wrapping
        """
        x, y = self.position

        if action == 0:  # up
            y = (y - 1) % grid_size[1]
        elif action == 1:  # down
            y = (y + 1) % grid_size[1]
        elif action == 2:  # left
            x = (x - 1) % grid_size[0]
        elif action == 3:  # right
            x = (x + 1) % grid_size[0]
        # action == 4: stay (no movement)

        self.position = (x, y)

    def update_energy(self, cost: float) -> None:
        """Update energy level."""
        self.energy -= cost
        self.age += 1

    def is_alive(self) -> bool:
        """Check if agent is still alive."""
        return self.energy > 0
