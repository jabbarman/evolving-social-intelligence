"""Brain module: Neural network implementation for agents."""

import numpy as np
from typing import Optional


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
