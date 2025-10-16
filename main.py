#!/usr/bin/env python3
"""Main entry point for the evolving social intelligence simulation."""

import yaml
import argparse
from pathlib import Path
from src.simulation import Simulation
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run the simulation."""
    parser = argparse.ArgumentParser(
        description="Evolving Social Intelligence Simulation"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Number of timesteps to run (overrides config)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    print(f"Grid size: {config['simulation']['grid_size']}")
    print(f"Initial population: {config['simulation']['initial_population']}")
    print(f"Population cap: {config['simulation']['population_cap']}")

    # Create simulation
    sim = Simulation(config)
    print(f"\nSimulation initialized with {len(sim.agents)} agents")

    # Determine number of steps
    max_steps = args.steps if args.steps else config['simulation']['max_timesteps']
    print(f"Running for {max_steps} timesteps...")

    # Run simulation
    try:
        for step in tqdm(range(max_steps), desc="Simulating"):
            sim.step()
            sim.timestep += 1

            # Print status every 10k steps
            if sim.timestep % 10000 == 0:
                print(f"\nTimestep {sim.timestep}:")
                print(f"  Population: {len(sim.agents)}")
                if len(sim.agents) > 0:
                    energies = [a.energy for a in sim.agents]
                    ages = [a.age for a in sim.agents]
                    print(f"  Avg energy: {sum(energies)/len(energies):.2f}")
                    print(f"  Avg age: {sum(ages)/len(ages):.2f}")

            # Check for extinction
            if len(sim.agents) == 0:
                print(f"\nPopulation extinct at timestep {sim.timestep}")
                break

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

    # Final statistics
    print(f"\n{'='*50}")
    print("Simulation completed!")
    print(f"Final timestep: {sim.timestep}")
    print(f"Final population: {len(sim.agents)}")

    # Save metrics
    sim.logger.save_metrics()
    print(f"Metrics saved to {sim.logger.save_dir}")


if __name__ == "__main__":
    main()
