import yaml
from pathlib import Path

from src.simulation import Simulation


def load_test_config(tmp_path):
    """Load the fast test config with a temporary logging directory."""
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "fast_test.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["logging"]["save_dir"] = str(tmp_path / "logs")
    return config


def test_simulation_runs_for_a_few_steps(tmp_path):
    """Basic smoke test ensuring the simulation can initialize and advance."""
    config = load_test_config(tmp_path)
    sim = Simulation(config)

    assert len(sim.agents) == config["simulation"]["initial_population"]
    sim.run(num_steps=3)

    assert sim.timestep == 3
    assert len(sim.agents) <= config["simulation"]["population_cap"]
    assert sim.environment.grid.shape == tuple(config["simulation"]["grid_size"])
