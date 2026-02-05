"""Visualization module: Pygame rendering for real-time visualization."""

import pygame
import numpy as np
from typing import Tuple, Dict, Optional

from src.analysis import calculate_behavioral_metrics


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
        self.COLOR_FOOD = (0, 200, 0)
        self.COLOR_AGENT_MIN = (50, 50, 255)  # Low energy (blue)
        self.COLOR_AGENT_MAX = (255, 50, 50)  # High energy (red)

        # Font for stats
        self.font = pygame.font.Font(None, 24)

    def render(self, environment, agents, timestep: int = 0, simulation_metrics: Dict[str, float] = None) -> bool:
        """Render current state of simulation.

        Args:
            environment: Environment instance
            agents: List of agents
            timestep: Current timestep
            simulation_metrics: Optional dict of social metrics from simulation
            
        Returns:
            False if user wants to quit, True otherwise

        Returns:
            continue_running: False if user closed window
        """
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False

        # Clear screen
        self.screen.fill(self.COLOR_EMPTY)

        # Draw food
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if environment.grid[x, y] == 1:
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(self.screen, self.COLOR_FOOD, rect)

        # Draw agents (colored by energy level)
        for agent in agents:
            x, y = agent.position

            # Color based on energy (interpolate between blue and red)
            energy_ratio = min(agent.energy / 150.0, 1.0)
            color = (
                int(self.COLOR_AGENT_MIN[0] + energy_ratio * (self.COLOR_AGENT_MAX[0] - self.COLOR_AGENT_MIN[0])),
                int(self.COLOR_AGENT_MIN[1] + energy_ratio * (self.COLOR_AGENT_MAX[1] - self.COLOR_AGENT_MIN[1])),
                int(self.COLOR_AGENT_MIN[2] + energy_ratio * (self.COLOR_AGENT_MAX[2] - self.COLOR_AGENT_MIN[2]))
            )

            center = (
                x * self.cell_size + self.cell_size // 2,
                y * self.cell_size + self.cell_size // 2
            )
            radius = max(2, self.cell_size // 3)
            pygame.draw.circle(self.screen, color, center, radius)

        # Draw stats overlay
        if len(agents) > 0:
            avg_energy = sum(a.energy for a in agents) / len(agents)
            avg_age = sum(a.age for a in agents) / len(agents)
            
            # Use simulation metrics if available, otherwise calculate basic ones
            if simulation_metrics:
                behavior = simulation_metrics
            else:
                behavior = calculate_behavioral_metrics(
                    agents,
                    getattr(environment, "food_energy_value", 0.0) if environment else 0.0,
                )

            stats_text = [
                f"Timestep: {timestep}",
                f"Population: {len(agents)}",
                f"Avg Energy: {avg_energy:.1f}",
                f"Avg Age: {avg_age:.0f}",
                f"Mean Dist: {behavior.get('mean_distance_per_step', 0):.2f}",
                f"Food Rate: {behavior.get('mean_food_discovery_rate', 0):.3f}",
                f"Entropy: {behavior.get('mean_movement_entropy', 0):.2f}",
                f"Comm Rate: {behavior.get('communication_rate', 0):.3f}",
                f"Signal Str: {behavior.get('mean_signal_strength', 0):.2f}",
                f"Transfers: {behavior.get('transfer_count', 0)}",
                f"Transfer Rate: {behavior.get('transfer_rate', 0):.3f}",
                f"Prox Bonus: {behavior.get('proximity_bonuses', 0):.1f}",
            ]

            y_offset = 10
            for text in stats_text:
                surface = self.font.render(text, True, (255, 255, 255))
                self.screen.blit(surface, (10, y_offset))
                y_offset += 25

        pygame.display.flip()
        self.clock.tick(self.fps)

        return True

    def close(self) -> None:
        """Close the visualization window."""
        pygame.quit()
