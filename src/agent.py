"""Agent module: Agent class with sensors and actuators."""

import math
from collections import deque
from itertools import count
from typing import Optional, Tuple

import numpy as np

from src.brain import Brain, SocialBrain


class Agent:
    """An agent that can perceive, decide, and act in the environment."""

    _id_counter = count()

    def __init__(self, position: Tuple[int, int], energy: float,
                 genome: Optional[np.ndarray] = None, brain: Optional[Brain] = None,
                 movement_history_length: int = 20,
                 agent_id: Optional[int] = None,
                 parent_id: Optional[int] = None,
                 generation: int = 0,
                 memory_state: Optional[np.ndarray] = None):
        """Initialize an agent.

        Args:
            position: (x, y) position in the grid
            energy: Initial energy level
            genome: Neural network weights (optional, random if not provided)
            brain: Brain instance (optional, created if not provided)
            movement_history_length: Number of recent actions to track for entropy
            agent_id: Optional explicit agent identifier (set when restoring/checkpointing)
            parent_id: Optional parent identifier for lineage tracking
            generation: Agent generation depth (0 for founders)
            memory_state: Optional memory state vector (16-dim, random if not provided)
        """
        self.position = position
        self.energy = energy
        self.age = 0
        self.id = agent_id if agent_id is not None else next(Agent._id_counter)
        self.parent_id = parent_id
        self.generation = generation
        self.offspring_count = 0
        self.lineage_root_id = self.id

        # Create brain if not provided
        if brain is None:
            self.brain = Brain()
        else:
            self.brain = brain

        # Set genome
        if genome is not None:
            self.brain.set_weights(genome)

        self.genome = self.brain.get_weights()
        self.movement_history_length = movement_history_length
        self.recent_actions = deque(maxlen=movement_history_length)
        self.last_move_distance = 0.0
        self.total_moves = 0
        self.food_discoveries = 0
        self.discovery_rate = 0.0
        self.movement_entropy = 0.0

        # Initialize memory state vector (16 dimensions for social recognition and temporal reasoning)
        if memory_state is not None:
            self.memory_state = memory_state.copy()
        else:
            self.memory_state = np.random.randn(16) * 0.1  # Small random initialization
            
        # Social brain state
        self.social_features_enabled = False
        self.communication_signal = 0.0
        self.transfer_willingness = 0.0

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

    def decide(self, observations: np.ndarray, signal_inputs: Optional[np.ndarray] = None) -> np.ndarray:
        """Process observations through neural network.

        Args:
            observations: 52-dimensional input vector
            signal_inputs: Optional 8-dimensional communication signals from nearby agents

        Returns:
            actions: 6-dimensional output (5 movement + 1 communication) for legacy brains,
                    or full social output for social brains
        """
        if self.social_features_enabled and isinstance(self.brain, SocialBrain) and not self.brain.legacy_mode:
            # Use social brain with memory and communication
            if signal_inputs is None:
                signal_inputs = np.zeros(8)  # No signals if not provided
                
            actions, new_memory, transfer_willingness = self.brain.forward_with_memory(
                observations, self.memory_state, signal_inputs
            )
            
            # Update internal state
            self.memory_state = new_memory
            
            # Normalize and store communication signal [-1, 1]
            raw_signal = float(actions[5])
            self.communication_signal = np.tanh(raw_signal)  # Bounded to [-1, 1]
            self.transfer_willingness = transfer_willingness
            
            return actions
        else:
            # Legacy brain mode - simple forward pass
            actions = self.brain.forward(observations)
            # Normalize communication signal for legacy brains too
            raw_signal = float(actions[5]) if len(actions) > 5 else 0.0
            self.communication_signal = np.tanh(raw_signal)  # Bounded to [-1, 1]
            self.transfer_willingness = 0.0
            return actions

    def move(self, action: int, grid_size: Tuple[int, int]) -> None:
        """Move the agent based on action (0=up, 1=down, 2=left, 3=right, 4=stay).

        Args:
            action: Movement action index
            grid_size: Size of the grid for toroidal wrapping
        """
        prev_x, prev_y = self.position
        x, y = prev_x, prev_y

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
        self._record_movement(prev_x, prev_y, x, y, grid_size, action)

    def update_energy(self, cost: float) -> None:
        """Update energy level."""
        self.energy -= cost
        self.age += 1

    def is_alive(self) -> bool:
        """Check if agent is still alive."""
        return self.energy > 0

    def enable_social_features(self) -> None:
        """Enable social features by converting to SocialBrain if needed."""
        if not self.social_features_enabled:
            if not isinstance(self.brain, SocialBrain):
                # Convert legacy Brain to SocialBrain
                self.brain = SocialBrain.from_legacy_brain(self.brain)
            
            # Set brain to social mode (non-legacy)
            if self.brain.legacy_mode:
                # Create new social brain with random weights
                old_activation = self.brain.activation_name
                self.brain = SocialBrain(legacy_mode=False, activation=old_activation)
                # Note: This creates new random weights - could implement weight migration later
                
            self.social_features_enabled = True
            # Update genome to reflect new brain weights
            self.genome = self.brain.get_weights()
            
    def get_communication_signal(self) -> float:
        """Get the current communication signal being emitted."""
        return self.communication_signal
        
    def get_transfer_willingness(self) -> float:
        """Get the current willingness to transfer resources."""
        return self.transfer_willingness
        
    def can_transfer_to(self, other_agent: 'Agent', transfer_amount: float = 10.0) -> bool:
        """Check if this agent can transfer energy to another agent.
        
        Args:
            other_agent: Target agent for transfer
            transfer_amount: Amount of energy to transfer
            
        Returns:
            True if transfer is possible (both agents willing and enough energy)
        """
        # Check if both agents are willing to transfer (threshold > 0.5 for willingness)
        self_willing = self.transfer_willingness > 0.5
        other_willing = other_agent.transfer_willingness > 0.5
        
        # Check if this agent has enough energy to transfer
        has_energy = self.energy > transfer_amount + 10  # Keep some buffer energy
        
        return self_willing and other_willing and has_energy
        
    def transfer_energy_to(self, other_agent: 'Agent', transfer_amount: float = 10.0) -> bool:
        """Transfer energy to another agent.
        
        Args:
            other_agent: Target agent
            transfer_amount: Amount of energy to transfer
            
        Returns:
            True if transfer was successful
        """
        if self.can_transfer_to(other_agent, transfer_amount):
            self.energy -= transfer_amount
            other_agent.energy += transfer_amount
            return True
        return False
        
    def get_communication_energy_cost(self, communication_cost: float = 0.5) -> float:
        """Calculate energy cost for communication signal emission.
        
        Args:
            communication_cost: Base energy cost for non-zero signals
            
        Returns:
            Energy cost based on signal strength
        """
        signal_strength = abs(self.communication_signal)
        if signal_strength > 0.01:  # Small threshold to avoid tiny costs
            return communication_cost * signal_strength
        return 0.0

    def record_food_discovery(self) -> None:
        """Register that the agent has discovered/consumed food."""
        self.food_discoveries += 1
        self._update_discovery_rate()

    def compute_movement_entropy(self) -> float:
        """Compute Shannon entropy of recent movement directions."""
        if not self.recent_actions:
            self.movement_entropy = 0.0
            return self.movement_entropy

        counts = np.bincount(list(self.recent_actions), minlength=5).astype(float)
        total = counts.sum()
        if total == 0:
            self.movement_entropy = 0.0
            return self.movement_entropy

        probabilities = counts[counts > 0] / total
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        self.movement_entropy = entropy
        return entropy

    def _record_movement(self, prev_x: int, prev_y: int,
                         new_x: int, new_y: int,
                         grid_size: Tuple[int, int], action: int) -> None:
        """Record movement statistics for behavioral metrics."""
        dx = self._wrapped_delta(prev_x, new_x, grid_size[0])
        dy = self._wrapped_delta(prev_y, new_y, grid_size[1])
        distance = math.hypot(dx, dy)
        self.last_move_distance = distance
        self.total_moves += 1
        self.recent_actions.append(action)
        self._update_discovery_rate()

    def _update_discovery_rate(self) -> None:
        """Update cached food discovery rate."""
        if self.total_moves == 0:
            self.discovery_rate = 0.0
        else:
            self.discovery_rate = self.food_discoveries / self.total_moves

    @staticmethod
    def _wrapped_delta(old: int, new: int, span: int) -> int:
        """Compute minimal wrapped delta for toroidal movement."""
        delta = new - old
        if abs(delta) > span // 2:
            if delta > 0:
                delta -= span
            else:
                delta += span
        return delta
