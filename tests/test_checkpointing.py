import numpy as np
import pytest
import yaml
from pathlib import Path

from src.simulation import Simulation


def _load_test_config(tmp_path):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "fast_test.yaml"
    config = yaml.safe_load(config_path.read_text())
    logs_dir = tmp_path / "logs"
    config["logging"]["save_dir"] = str(logs_dir)
    config["logging"]["checkpoint_interval"] = 0
    return config


def test_checkpoint_roundtrip(tmp_path):
    config = _load_test_config(tmp_path)
    sim = Simulation(config)

    for _ in range(3):
        sim.step()
        sim.timestep += 1

    checkpoint_path = sim.save_checkpoint()
    lineage_db_path = Path(sim.logger.save_dir) / "lineage.db"
    assert lineage_db_path.exists()

    resumed = Simulation.from_checkpoint(checkpoint_path)

    assert resumed.timestep == sim.timestep
    assert len(resumed.agents) == len(sim.agents)
    assert np.array_equal(resumed.environment.grid, sim.environment.grid)
    resumed_lineage_db = Path(resumed.logger.save_dir) / "lineage.db"
    assert resumed_lineage_db.exists()
    assert resumed_lineage_db == lineage_db_path

    for original, restored in zip(sim.agents, resumed.agents):
        assert original.position == restored.position
        assert restored.energy == pytest.approx(original.energy)
        assert restored.age == original.age
        assert np.allclose(restored.genome, original.genome)
        assert original.id == restored.id
        assert original.parent_id == restored.parent_id
        assert original.generation == restored.generation
        assert original.lineage_root_id == restored.lineage_root_id
        assert original.total_moves == restored.total_moves
        assert original.food_discoveries == restored.food_discoveries
        assert restored.discovery_rate == pytest.approx(original.discovery_rate)
        assert list(original.recent_actions) == list(restored.recent_actions)


def test_checkpoint_restores_rng_state(tmp_path):
    config = _load_test_config(tmp_path)
    sim = Simulation(config)

    sim.step()
    sim.timestep += 1
    checkpoint_path = sim.save_checkpoint()

    expected_random = np.random.random()

    resumed = Simulation.from_checkpoint(checkpoint_path)

    actual_random = np.random.random()
    assert actual_random == pytest.approx(expected_random)

    # Ensure resumed simulation continues to run
    resumed.step()
    resumed.timestep += 1
