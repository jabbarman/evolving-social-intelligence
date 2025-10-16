"""Visualization module: Pygame rendering for real-time visualization."""

import pygame
import numpy as np
from typing import Tuple


class Visualizer:
    """Real-time visualization using Pygame."""

    def __init__(self, grid_size: Tuple[int, int], cell_size: int = 8,
                 fps: int = 30):
        """Initialize the visualizer.

        Args:
            grid_size: (width, height) of the simulation grid
            cell_size: Size of each cell in pixels
            fps: Target frames per second
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps

        # Calculate window size
        self.window_size = (grid_size[0] * cell_size, grid_size[1] * cell_size)

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Evolving Social Intelligence")
        self.clock = pygame.time.Clock()

        # Colors
        self.COLOR_EMPTY = (0, 0, 0)
        self.COLOR_FOOD = (0, 255, 0)
        self.COLOR_AGENT = (255, 255, 255)

    def render(self, environment, agents) -> bool:
        """Render current state of simulation.

        Args:
            environment: Environment instance
            agents: List of agents

        Returns:
            continue_running: False if user closed window
        """
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Clear screen
        self.screen.fill(self.COLOR_EMPTY)

        # TODO: Draw food
        # TODO: Draw agents

        pygame.display.flip()
        self.clock.tick(self.fps)

        return True

    def close(self) -> None:
        """Close the visualization window."""
        pygame.quit()
