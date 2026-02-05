#!/usr/bin/env python3
"""Main entry point for the evolving social intelligence simulation."""

import yaml
import argparse
from pathlib import Path
from src.simulation import Simulation
from src.visualization import Visualizer
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
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Resume simulation state from a checkpoint file'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create simulation
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        logging_override = {"logging": config["logging"]} if "logging" in config else None
        sim = Simulation.from_checkpoint(args.resume_from, config_override=logging_override)
        print(f"Checkpoint configuration restored (override source: {args.config})")
    else:
        sim = Simulation(config)
        print(f"Loaded config from {args.config}")

    config = sim.config
    print(f"Grid size: {config['simulation']['grid_size']}")
    print(f"Initial population: {config['simulation']['initial_population']}")
    print(f"Population cap: {config['simulation']['population_cap']}")
    print(f"\nSimulation initialized with {len(sim.agents)} agents at timestep {sim.timestep}")

    # Determine number of steps
    max_steps = args.steps if args.steps else config['simulation']['max_timesteps']
    print(f"Running until timestep {max_steps} (current: {sim.timestep})...")

    # Initialize visualizer if requested
    viz = None
    if not args.no_viz:
        grid_size = tuple(config['simulation']['grid_size'])
        cell_size = 8 if grid_size[0] <= 100 else 4
        viz = Visualizer(grid_size, cell_size=cell_size, fps=30)
        print("Visualization enabled (press ESC to close window)")

    # Run simulation
    try:
        if viz:
            # Run with visualization
            running = True
            while sim.timestep < max_steps and running:
                sim.step()
                sim.timestep += 1
                sim._maybe_save_checkpoint()

                # Render visualization with real-time metrics
                current_metrics = sim.get_current_behavioral_metrics()
                running = viz.render(sim.environment, sim.agents, sim.timestep, current_metrics)

                # Print status every 1k steps
                if sim.timestep % 1000 == 0:
                    print(f"Timestep {sim.timestep}: Population={len(sim.agents)}")

                # Check for extinction
                if len(sim.agents) == 0:
                    print(f"\nPopulation extinct at timestep {sim.timestep}")
                    break

        else:
            # Run without visualization (faster)
            if sim.timestep >= max_steps:
                print("Simulation has already reached the requested timestep count.")
            else:
                for _ in tqdm(range(sim.timestep, max_steps), desc="Simulating",
                              initial=sim.timestep, total=max_steps):
                    sim.step()
                    sim.timestep += 1
                    sim._maybe_save_checkpoint()

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
        try:
            checkpoint_path = sim.save_checkpoint()
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to save checkpoint on interrupt: {exc}")
    finally:
        if viz:
            viz.close()
        if sim.lineage_tracker:
            sim.lineage_tracker.close()

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
