import pytest

from src.agent import Agent
from src.environment import Environment


def test_agent_perception_and_movement_wraparound():
    """Agents should perceive locally and wrap around grid boundaries when moving."""
    env = Environment(grid_size=(5, 5), food_spawn_rate=0.0, food_energy_value=1, seed=123)
    env.grid[0, 0] = 1  # place food at the agent's position

    agent = Agent(position=(0, 0), energy=10)
    neighbor = Agent(position=(1, 0), energy=10)

    observations = agent.perceive(env, [agent, neighbor])

    assert observations.shape == (52,)
    assert observations[-2] == pytest.approx(agent.energy / 100.0)
    assert observations[-1] == pytest.approx(0.0)

    agent.move(action=2, grid_size=env.grid_size)  # move left with wraparound
    assert agent.position == (4, 0)
