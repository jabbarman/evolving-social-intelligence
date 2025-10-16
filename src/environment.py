"""Environment module: World, resources, and physics."""

import numpy as np
from typing import Tuple


class Environment:
    """2D grid-based world with resources and physics."""

    def __init__(self, grid_size: Tuple[int, int], food_spawn_rate: float,
                 food_energy_value: float, seed: int = None):
        """Initialize the environment.

        Args:
            grid_size: (width, height) of the grid
            food_spawn_rate: Probability of food spawning per cell per timestep
            food_energy_value: Energy value of each food unit
            seed: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.food_spawn_rate = food_spawn_rate
        self.food_energy_value = food_energy_value

        if seed is not None:
            np.random.seed(seed)

        # Grid: 0 = empty, 1 = food
        self.grid = np.zeros(grid_size, dtype=np.int8)

    def spawn_food(self) -> None:
        """Spawn food probabilistically in empty cells."""
        # Generate random probabilities for each cell
        spawn_mask = np.random.random(self.grid_size) < self.food_spawn_rate

        # Only spawn in empty cells (where grid == 0)
        empty_mask = self.grid == 0

        # Set food where both conditions are true
        self.grid[spawn_mask & empty_mask] = 1

    def get_cell(self, position: Tuple[int, int]) -> int:
        """Get the contents of a cell (with toroidal wrapping)."""
        x, y = position
        x = x % self.grid_size[0]
        y = y % self.grid_size[1]
        return self.grid[x, y]

    def consume_food(self, position: Tuple[int, int]) -> float:
        """Consume food at a position and return energy gained."""
        x, y = position
        x = x % self.grid_size[0]
        y = y % self.grid_size[1]

        if self.grid[x, y] == 1:
            self.grid[x, y] = 0  # Remove food
            return self.food_energy_value
        return 0.0

    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
