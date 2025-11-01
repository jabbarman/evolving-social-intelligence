"""Evolution module: Reproduction, mutation, and selection."""

import numpy as np
from typing import List
from src.agent import Agent


class Evolution:
    """Handles evolutionary mechanics: reproduction, mutation, selection."""

    def __init__(self, mutation_rate: float, mutation_std: float):
        """Initialize evolution parameters.

        Args:
            mutation_rate: Fraction of weights to mutate
            mutation_std: Standard deviation of mutation noise
        """
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std

    def reproduce(self, parent: Agent, reproduction_cost: float,
                  offspring_position: tuple, movement_history_length: int = 20) -> Agent:
        """Create offspring from parent with mutations.

        Args:
            parent: Parent agent
            reproduction_cost: Energy cost of reproduction
            offspring_position: Position for the new offspring
            movement_history_length: Recent action buffer size for offspring

        Returns:
            offspring: New agent with mutated genome
        """
        # Copy and mutate parent genome
        offspring_genome = self.mutate(parent.genome.copy())

        # Create offspring with mutated genome
        offspring = Agent(
            position=offspring_position,
            energy=reproduction_cost,  # Offspring gets the energy from parent
            genome=offspring_genome,
            movement_history_length=movement_history_length,
        )

        return offspring

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """Apply mutations to genome.

        Args:
            genome: Parent genome (neural network weights)

        Returns:
            mutated_genome: Mutated copy of genome
        """
        # Create mutation mask (which weights to mutate)
        mutation_mask = np.random.random(genome.shape) < self.mutation_rate

        # Generate Gaussian noise
        noise = np.random.normal(0, self.mutation_std, genome.shape)

        # Apply mutations only where mask is True
        genome[mutation_mask] += noise[mutation_mask]

        return genome

    def select(self, agents: List[Agent], population_cap: int) -> List[Agent]:
        """Apply selection pressure to maintain population cap.

        Args:
            agents: List of agents
            population_cap: Maximum population size

        Returns:
            selected_agents: Agents that survive selection
        """
        if len(agents) <= population_cap:
            return agents

        # Sort by age (oldest first) and remove oldest
        agents_sorted = sorted(agents, key=lambda a: a.age, reverse=True)
        num_to_remove = len(agents) - population_cap

        # Return younger agents
        return agents_sorted[num_to_remove:]
