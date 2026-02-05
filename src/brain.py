"""Brain module: Neural network implementation for agents."""

import numpy as np
from typing import Optional, Tuple


class Brain:
    """Simple feedforward neural network for agent decision-making using NumPy."""

    def __init__(self, input_size: int = 52, hidden_size: int = 32,
                 output_size: int = 6, activation: str = "relu"):
        """Initialize the neural network.

        Args:
            input_size: Number of input neurons
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            activation: Activation function name
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation

        # Initialize weights with Xavier initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def _tanh(self, x: np.ndarray) -> np.ndarray:
        """Tanh activation function."""
        return np.tanh(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Args:
            x: Input array of shape (input_size,)

        Returns:
            output: Output array of shape (output_size,)
        """
        # Hidden layer
        hidden = x @ self.w1 + self.b1

        # Activation
        if self.activation_name == "relu":
            hidden = self._relu(hidden)
        elif self.activation_name == "tanh":
            hidden = self._tanh(hidden)

        # Output layer
        output = hidden @ self.w2 + self.b2

        return output

    def get_weights(self) -> np.ndarray:
        """Get all network weights as a flat array."""
        return np.concatenate([
            self.w1.flatten(),
            self.b1.flatten(),
            self.w2.flatten(),
            self.b2.flatten()
        ])

    def set_weights(self, weights: np.ndarray) -> None:
        """Set network weights from a flat array."""
        idx = 0

        # w1
        w1_size = self.input_size * self.hidden_size
        self.w1 = weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size

        # b1
        self.b1 = weights[idx:idx + self.hidden_size]
        idx += self.hidden_size

        # w2
        w2_size = self.hidden_size * self.output_size
        self.w2 = weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size

        # b2
        self.b2 = weights[idx:idx + self.output_size]

    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return (self.input_size * self.hidden_size + self.hidden_size +
                self.hidden_size * self.output_size + self.output_size)


class SocialBrain(Brain):
    """Enhanced neural network with memory state and social I/O for agents."""
    
    def __init__(self, legacy_mode: bool = False, activation: str = "relu"):
        """Initialize the social neural network.
        
        Args:
            legacy_mode: If True, use original 52→32→6 architecture for compatibility
            activation: Activation function name
        """
        self.legacy_mode = legacy_mode
        
        if legacy_mode:
            # Original architecture for backward compatibility
            super().__init__(input_size=52, hidden_size=32, output_size=6, activation=activation)
        else:
            # New social architecture: 76→48→23
            # Inputs: 52 (environment) + 16 (memory) + 8 (signals) = 76
            # Outputs: 6 (movement+comm) + 16 (memory update) + 1 (transfer) = 23
            super().__init__(input_size=76, hidden_size=48, output_size=23, activation=activation)
            
    def forward_with_memory(self, environment_obs: np.ndarray, memory_state: np.ndarray, 
                           signal_inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Forward pass with memory and social inputs.
        
        Args:
            environment_obs: Environmental observations (52-dim)
            memory_state: Current memory state (16-dim)  
            signal_inputs: Communication signals from nearby agents (8-dim)
            
        Returns:
            actions: Movement and communication actions (6-dim)
            new_memory: Updated memory state (16-dim)
            transfer_willingness: Resource transfer willingness [0,1]
        """
        if self.legacy_mode:
            # Legacy mode: only use environment observations
            actions = super().forward(environment_obs)
            # Return unchanged memory and no transfer willingness
            return actions, memory_state, 0.0
            
        # Combine all inputs
        full_input = np.concatenate([environment_obs, memory_state, signal_inputs])
        
        # Forward pass
        full_output = super().forward(full_input)
        
        # Split outputs
        actions = full_output[:6]  # Movement (5) + communication signal (1)
        new_memory = np.tanh(full_output[6:22])  # Memory update (16) - bounded
        transfer_willingness = float(1.0 / (1.0 + np.exp(-full_output[22])))  # Sigmoid for [0,1]
        
        return actions, new_memory, transfer_willingness
        
    @classmethod  
    def from_legacy_brain(cls, legacy_brain: Brain) -> 'SocialBrain':
        """Create a SocialBrain from an existing Brain for migration.
        
        Args:
            legacy_brain: Existing Brain instance
            
        Returns:
            SocialBrain instance in legacy mode with copied weights
        """
        social_brain = cls(legacy_mode=True, activation=legacy_brain.activation_name)
        social_brain.set_weights(legacy_brain.get_weights())
        return social_brain
