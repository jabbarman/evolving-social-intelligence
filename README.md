# Evolving Social Intelligence

An open-ended evolutionary simulation where AI agents develop intelligence through social interaction, recapitulating the environmental and social pressures that shaped biological intelligence on Earth.

## What This Is

This project creates a minimal but complete artificial life simulation where:

- **Neural network agents** with evolved brains (legacy: 52→32→6, social: 76→48→23) perceive their environment and make decisions
- **Evolution drives adaptation** through asexual reproduction with mutation, natural selection, and energy-based survival  
- **Social behaviors emerge** from agents evolving communication, cooperation, and memory systems
- **Emergent intelligence** arises from agents learning to forage, communicate, cooperate, and form social structures

The simulation runs on a 2D toroidal grid where agents must:
- Find and consume food to maintain energy
- Manage metabolic costs of movement, communication, and survival
- Form social groups for proximity benefits and resource sharing
- Reproduce when they gather enough energy
- Pass evolved neural networks and memory patterns to offspring

**Core Philosophy**: Rather than engineering intelligent behaviors, we provide agents with social capabilities and let evolution discover which strategies work through millions of timesteps of trial and error.

## Key Features

### Core Simulation
- **Fast NumPy-based simulation**: 150+ timesteps/second without visualization
- **Real-time Pygame visualization**: Watch evolution happen with color-coded agents and live social metrics
- **Configurable parameters**: Easy YAML configs for all simulation aspects
- **Comprehensive checkpointing**: Save/resume simulations with full state preservation

### Social Behavior System ✨
- **Memory**: 16-dimensional persistent memory state that evolves across generations
- **Communication**: Energy-costly signaling system with perception by nearby agents  
- **Cooperation**: Bilateral resource transfer requiring mutual consent
- **Social Clustering**: Proximity benefits encourage group formation
- **Selection Pressure**: Resource scarcity drives evolution of social strategies

### Advanced Analytics
- **Real-time social metrics**: Communication rates, transfer events, clustering patterns
- **Behavioral tracking**: Movement efficiency, food discovery, social interaction patterns
- **Lineage analysis**: Ancestry tracking with genetic distance and diversity indices  
- **Social network analysis**: Dynamic relationship mapping and community detection

## Project Status

**Phase 2: Social Behaviors Complete** ✅ 

Current capabilities:
- ✅ **Memory systems**: Agents maintain persistent 16-dim memory across timesteps
- ✅ **Communication**: Agents emit and perceive signals with energy costs and neural processing
- ✅ **Cooperation**: Bilateral resource transfers between willing agents  
- ✅ **Social clustering**: Proximity benefits encourage group formation
- ✅ **Selection pressure**: Configurable scarcity drives social evolution
- ✅ **Real-time visualization**: Live display of social metrics during simulation
- ✅ **Complete neural architecture**: Backward-compatible social brains (76→48→23)

**Active Research**: Long-term experiments showing evolution of proto-languages, cooperation strategies, and social hierarchies.

**Future Directions**: Advanced social analytics, environmental complexity, multi-species dynamics, open-ended evolution.

## Quick Demo

### Social Evolution in Action

![Social Behavior Demo](docs/social-evolution-demo.gif)

*Agents with social intelligence: clustering for proximity bonuses, communicating via signals, and sharing resources through cooperation. Watch the real-time social metrics (top-left) as behaviors evolve.*

**What you're seeing:**
- **Colored circles**: Agents with evolving social behaviors (blue=low energy, red=high energy)
- **Green squares**: Food resources agents compete for and share information about
- **Social metrics** (top-left): Live tracking of communication rates, resource transfers, and clustering behavior
- **Dynamic evolution**: Social strategies emerging over time through natural selection

### Run Your Own Demo

**Start social evolution experiment:**
```bash
python3 main.py --config configs/social_evolution.yaml --steps 2000
```

**Quick test with social behaviors:**
```bash  
python3 main.py --config configs/social_test.yaml --steps 500
```

**Create additional animations:** Follow the [demo creation guide](docs/create-animated-demo.md)

> **💡 Tip**: Social behaviors emerge gradually over 500-2000+ timesteps as agents evolve communication and cooperation strategies through natural selection. The animation shows accelerated footage of this evolutionary process!

---
## System Dependencies

Before installing Python dependencies, you need to install SDL2 libraries:

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev
```

### macOS (with Homebrew):
```bash
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf freetype
```

### Fedora/RHEL:
```bash
sudo dnf install python3-devel SDL2-devel SDL2_image-devel SDL2_mixer-devel SDL2_ttf-devel freetype-devel
```


## Setup and Installation

### Prerequisites

- Python 3.10 or higher (3.13 recommended)
- pip package manager
- ~50MB disk space for dependencies

### Installation Steps

1. **Clone or download this repository**:
```bash
git clone https://github.com/jabbarman/evolving-social-intelligence.git
cd evolving-social-intelligence
```

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This installs:
- `numpy` - Fast numerical computing
- `pygame` - Real-time visualization
- `matplotlib` - Plotting and analysis
- `pandas` - Data manipulation
- `pyyaml` - Configuration files
- `tqdm` - Progress bars

**Note**: PyTorch is listed in requirements but optional. The project uses NumPy for neural networks due to Python 3.13 compatibility. If you're on Python 3.11 or earlier and want GPU acceleration, you can install PyTorch separately, but it's not required.

### Verify Installation

Run a quick test to make sure everything works:
```bash
python3 main.py --config configs/fast_test.yaml --steps 100 --no-viz
```

You should see output like:
```
Loaded config from configs/fast_test.yaml
Grid size: [50, 50]
Initial population: 20
Population cap: 100

Simulation initialized with 20 agents
Running for 100 timesteps...
Simulating: 100%|██████████| 100/100 [00:00<00:00, 150.23it/s]

==================================================
Simulation completed!
Final timestep: 100
Final population: 65
Metrics saved to experiments/logs
```

---

## Running Simulations

### Basic Usage

**Run with social behaviors and visualization**:
```bash  
python3 main.py --config configs/social_evolution.yaml
```

**Watch social evolution in real-time**:
```bash
python3 main.py --config configs/social_evolution.yaml --steps 5000
```

**Fast run without visualization**:
```bash
python3 main.py --config configs/social_evolution.yaml --no-viz --steps 10000
```

**Standard non-social simulation**:
```bash
python3 main.py --config configs/default.yaml
```

### Social Behavior Configurations

For studying social evolution, use configurations with selection pressure:

**Quick social test** (20 agents, social features enabled):
```bash
python3 main.py --config configs/social_test.yaml --steps 1000
```

**Full social evolution** (resource scarcity drives cooperation):
```bash
python3 main.py --config configs/social_evolution.yaml --steps 50000
```

### Checkpointing and Resuming Runs

- The simulation periodically writes compressed checkpoints to `experiments/logs/checkpoints/` (or the `logging.save_dir` you configure). The cadence is set by `logging.checkpoint_interval`; set it to `0` to disable automatic saves.
- Hitting `Ctrl+C` during a run also triggers a best-effort checkpoint before the program exits, so long-running experiments can stop safely.
- Resume from any checkpoint file with the new `--resume-from` flag:

```bash
python3 main.py --resume-from experiments/logs/checkpoints/checkpoint_00050000.pkl.gz --no-viz
```

The configuration stored in the checkpoint is restored automatically. Values under `logging` can still be overridden by the config you pass via `--config`.

### Available Configurations

- `configs/default.yaml` - Standard setup (100×100 grid, 50 agents) with social features disabled
- `configs/fast_test.yaml` - Quick testing (50×50 grid, 20 agents) for development
- `configs/social_test.yaml` - Social features enabled with standard parameters
- `configs/social_evolution.yaml` - **Recommended**: Resource scarcity + social benefits for evolution

### Creating Custom Configs

Copy and modify an existing config file:

```yaml
# Example social evolution configuration
simulation:
  grid_size: [80, 80]           # Smaller grid = more interaction
  population_cap: 150           # Lower cap = more competition  
  initial_population: 30        # Start smaller for selection pressure

environment:
  food_spawn_rate: 0.005        # Scarcity drives cooperation
  food_energy_value: 15         # Higher value rewards finding food

agent:
  base_metabolic_cost: 1.2      # Higher survival pressure
  communication_cost: 0.3       # Affordable communication
  transfer_amount: 15.0         # Beneficial cooperation amounts
  proximity_bonus: 2.0          # Energy reward for social clustering
  proximity_range: 1            # Range for proximity benefits

brain:
  social_features: true         # Enable social behaviors
  input_size: 76               # Auto-configured for social brains
  output_size: 23              # Actions + memory + cooperation

behavioral_metrics:
  log_interval: 50             # More frequent social metric updates
```

The `behavioral_metrics` block controls movement tracking and food discovery aggregation, while `lineage_tracking` governs how often ancestry summaries (`lineage_stats.json`, `lineage.db`) are written.

### Understanding the Visualization

When running with visualization enabled:

- **Black background**: Empty space
- **Green squares**: Food resources  
- **Colored circles**: Agents
  - Blue = Low energy (near starvation)
  - Purple/Pink = Medium energy
  - Red = High energy (ready to reproduce)
- **Real-time social metrics** (top-left overlay):
  - Timestep and population counters
  - Energy and age statistics  
  - Movement patterns and food rates
  - **Communication rate**: Frequency of signal emission
  - **Signal strength**: Average communication intensity
  - **Transfer count**: Resource sharing events per interval
  - **Transfer rate**: Cooperation frequency per agent
  - **Proximity bonuses**: Energy gained from social clustering

**Controls**:
- Press `ESC` or close window to stop simulation
- Terminal shows periodic status updates with social behavior statistics

## Social Behavior System

### Agent Capabilities

**Memory System**:
- 16-dimensional persistent memory state
- Inherited from parents with mutations during reproduction  
- Updated each timestep via neural network outputs
- Enables recognition, temporal reasoning, and learning

**Communication**:
- Agents emit signals via 6th neural output (normalized to [-1,1])  
- Energy cost: 0.5 × |signal strength| per timestep
- Signals perceived by agents within 5×5 perception grid
- 8-dimensional signal input vector processed by social brain

**Cooperation**:
- Bilateral resource transfer requiring mutual consent (>0.5 willingness)
- Agents must be adjacent (within 1 cell) to transfer energy
- Configurable transfer amount (default: 10-15 energy units)
- Transfer events tracked for behavioral analytics

**Social Clustering**:
- Proximity bonuses reward agents for staying near others
- Energy bonus: 2.0 × nearby_agents × 0.5 (diminishing returns)
- Encourages group formation and social organization
- Balances clustering benefits vs overcrowding costs

### Neural Architecture

**Legacy Brain** (backward compatibility):
- 52 inputs → 32 hidden → 6 outputs (~1,700 parameters)
- Environmental perception only (food + agents + energy + age)

**Social Brain** (new capabilities):
- 76 inputs → 48 hidden → 23 outputs (~4,800 parameters)
- **Inputs**: 52 environmental + 16 memory + 8 communication signals
- **Outputs**: 6 actions + 16 memory update + 1 transfer willingness
- Full backward compatibility with automatic conversion

### Evolution of Social Behaviors

**Selection Pressure**:
- Resource scarcity makes cooperation advantageous
- Communication costs must be offset by survival benefits
- Memory allows agents to recognize and remember interactions
- Proximity benefits reward social agents over solitary ones

**Emergent Patterns**:
- Signal evolution: Random → meaningful communication patterns
- Cooperation networks: Reciprocal altruism and kin selection
- Memory specialization: Recognition systems and behavioral templates  
- Social hierarchies: Leadership, following, and group coordination

## Behavioral Metrics

The simulation records comprehensive behavioral and social metrics:

**Basic Behaviors**:
- **Movement patterns**: Distance per step, movement entropy, exploration efficiency
- **Foraging success**: Food discovery rates, consumption patterns, energy management
- **Population dynamics**: Births, deaths, age distributions, lineage tracking

**Social Behaviors**:
- **Communication metrics**: Signal emission rates, signal strength evolution, energy costs
- **Cooperation tracking**: Transfer events, reciprocal altruism patterns, resource sharing networks
- **Social clustering**: Proximity bonuses, group formation, spatial organization patterns
- **Memory evolution**: Memory state divergence, specialization patterns, inheritance dynamics

**Advanced Analytics**:
- **Social network analysis**: Dynamic relationship graphs, centrality measures, community detection
- **Behavioral synchronization**: Coordinated movement, collective decision-making patterns  
- **Signal semantics**: Communication pattern analysis, proto-language emergence detection
- **Evolutionary pressure**: Selection gradients, fitness landscapes, adaptive radiations

All metrics are written to a compressed archive at `experiments/logs/metrics.npz`, while lineage summaries live in `experiments/logs/lineage_stats.json` and the SQLite database `experiments/logs/lineage.db`. Explore them with the new `notebooks/behavioral_analysis.ipynb` notebook or your favorite analysis tools (or connect to the SQLite database directly for richer queries).

For long runs, the helper script `scripts/plot_behavioral_trends.py` reads the NumPy archive and produces a down-sampled trend plot:

```bash
python scripts/plot_behavioral_trends.py --logs-dir experiments/logs --stride 5000
```

The plots are written to `experiments/logs/plots/`.

Lineage summaries are smaller but you can turn them into quick visuals with:

```bash
python scripts/plot_lineage_dynamics.py --logs-dir experiments/logs
```

This emits `lineage_metrics_summary.png` plus a bar chart of the latest dominant lineages inside `experiments/logs/plots/`.

---

## Project Structure

```
evolving-social-intelligence/
├── README.md                    # This file  
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── main.py                      # Entry point
├── configs/                     # Configuration files
│   ├── default.yaml            # Standard non-social simulation
│   ├── fast_test.yaml          # Quick testing configuration
│   ├── social_test.yaml        # Social features enabled
│   └── social_evolution.yaml   # Resource scarcity + social evolution
├── src/                         # Source code
│   ├── __init__.py
│   ├── environment.py           # Grid world, food spawning
│   ├── agent.py                 # Agent sensors, memory, social behaviors
│   ├── brain.py                 # Neural networks (legacy + social)
│   ├── evolution.py             # Reproduction, mutation, inheritance
│   ├── simulation.py            # Main loop with social mechanics
│   ├── visualization.py         # Pygame rendering + social metrics
│   ├── analysis.py              # Comprehensive behavioral metrics
│   └── lineage.py              # Lineage tracking and ancestry analysis
├── experiments/
│   └── logs/                    # Metrics, checkpoints, social analytics
├── tests/                       # Unit tests
└── docs/                        # Documentation
```

---

## Understanding the Simulation

### Agent Architecture

Each agent has:
- **Position** on the 2D toroidal grid
- **Energy** level (dies when ≤ 0)  
- **Age** in timesteps
- **Memory state** (16-dimensional persistent vector)
- **Social capabilities** (communication signal, transfer willingness)
- **Neural network brain** with 1,700-4,800 parameters (legacy vs social)

**Environmental Perception** (52 inputs):
- 5×5 grid of food locations (25 values)
- 5×5 grid of nearby agents (25 values)  
- Own energy level (normalized)
- Own age (normalized)

**Social Perception** (additional 24 inputs for social brains):
- Memory state from previous timestep (16 values)
- Communication signals from nearby agents (8 values)

**Actions** (6-23 outputs):
- 5 movement actions (up, down, left, right, stay) - softmax selection
- 1 communication signal (tanh normalized to [-1,1])
- **Social brains also output**:
  - 16 memory state updates
  - 1 transfer willingness value [0,1]

### Simulation Loop

Each timestep:
1. **Perception**: All agents observe local 5×5 area + social signals
2. **Decision**: Neural networks process observations + memory → actions + memory update
3. **Communication**: Signal emission with energy costs, signal gathering for perception  
4. **Movement**: Agents move based on network outputs with energy costs
5. **Social interactions**: Resource transfers between willing adjacent agents
6. **Proximity benefits**: Energy bonuses for agents near others (social clustering)
7. **Consumption**: Agents on food cells gain energy  
8. **Metabolism**: All agents lose base metabolic energy
9. **Reproduction**: High-energy agents create mutated offspring with inherited memory
10. **Death**: Agents with energy ≤ 0 are removed
11. **Selection**: Population cap enforcement (oldest agents removed first)
12. **Environment**: New food spawns probabilistically
13. **Logging**: Social and behavioral metrics recorded at intervals

### Evolutionary Mechanics

**Reproduction**:
- Triggered when agent energy > reproduction threshold (130-150)
- Parent loses reproduction cost (60-80 energy), offspring gets same amount
- Offspring placed in adjacent empty cell with inherited memory state
- Offspring genome = parent genome + mutations + memory inheritance

**Mutation**:
- 10-12% of neural network weights are mutated (configurable)
- Gaussian noise added: N(0, 0.08-0.1) for fine-tuning
- Memory state inherits with small random perturbations  
- Allows exploration of strategy and memory patterns

**Selection Pressure**:
- **Natural selection**: agents die when energy depletes
- **Resource competition**: scarce food rewards efficient foragers
- **Social advantages**: communication and cooperation provide survival benefits
- **Population pressure**: oldest agents removed if over capacity
- **Energy economics**: communication costs balanced against cooperation benefits

---

## Analyzing Results

### Notebook Exploration

The repository includes `notebooks/behavioral_analysis.ipynb`, a Jupyter Notebook that walks through quick diagnostics and custom lineage queries. Jupyter notebooks let you mix Python code, narrative text, and plots in a single document—perfect for iterating on analysis without writing standalone scripts. See the [Jupyter documentation](https://docs.jupyter.org/en/stable/) or the [Project Jupyter site](https://jupyter.org/) to learn more.

The bundled notebook currently demonstrates:

- Loading behavioral metrics from `metrics.npz`.
- Plotting population dynamics over time.
- Visualizing dominant lineage share across checkpoints.
- Inspecting descendants of a specific founder via the SQLite lineage database.

To use it:

1. Run a simulation (for example `python3 main.py --config configs/default.yaml --no-viz`) so metrics and lineage files populate `experiments/logs`.
2. Launch Jupyter (`jupyter notebook` or `jupyter lab`) from the repository root and open `notebooks/behavioral_analysis.ipynb`.
3. Execute the cells top-to-bottom, editing parameters like `FOUNDER_ID` as needed. The notebook will render inline charts and tables based on your latest run.



Metrics are saved to `experiments/logs/metrics.npz` after each run:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load metrics
with np.load('experiments/logs/metrics.npz') as data:
    timesteps = data['timesteps']
    population = data['population']
    mean_energy = data['mean_energy']
    mean_age = data['mean_age']

# Plot population over time
plt.plot(timesteps, population)
plt.xlabel('Timestep')
plt.ylabel('Population')
plt.title('Population Dynamics')
plt.show()

# Plot energy and age
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(timesteps, mean_energy)
ax1.set_title('Mean Energy')
ax2.plot(timesteps, mean_age)
ax2.set_title('Mean Age')
plt.show()
```

Tracked metrics:
- **Population dynamics**: `timesteps`, `population`, `births`, `deaths`
- **Energy and lifespan**: `mean_energy`, `mean_age`, `max_age`
- **Environmental**: `total_food` availability
- **Movement patterns**: `mean_distance_per_step`, `movement_entropy`
- **Foraging**: `mean_food_discovery_rate`, `total_food_consumed`
- **Social behaviors**: 
  - `communication_rate` - Signal emission frequency per agent
  - `mean_signal_strength` - Average communication intensity  
  - `transfer_count` - Resource sharing events per logging interval
  - `transfer_rate` - Cooperation frequency per agent per timestep
  - `proximity_bonuses` - Total energy gained from social clustering

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'pygame'"**
- Run `pip install -r requirements.txt` to install dependencies

**Simulation runs very slowly**
- Use `--no-viz` flag to disable visualization (10x faster)
- Reduce grid size or population in config
- Check that numpy is using optimized BLAS libraries

**Population goes extinct quickly**
- Use `configs/social_evolution.yaml` for balanced resource scarcity
- Increase `food_spawn_rate` for less competitive environments (try 0.01)
- Lower `base_metabolic_cost` (try 1.0) for easier survival
- Increase `initial_population` to improve survival chances

**Social behaviors aren't emerging**
- Ensure `brain.social_features: true` in configuration
- Use `configs/social_evolution.yaml` for proper selection pressure
- Run longer simulations (10k+ timesteps) for evolutionary time scales
- Check that `proximity_bonus` > 0 to reward social clustering

**"TypeError: Object of type int64 is not JSON serializable"**
- This should be fixed in the latest version
- If you encounter this, update `src/analysis.py` line 90-91

**Visualization window doesn't open**
- Check pygame is installed: `python3 -c "import pygame; print(pygame.__version__)"`
- Try running with `--no-viz` to see if core simulation works
- On headless servers, visualization won't work (use `--no-viz`)

---

## Contributing

This is an open research project and contributions are welcome! Areas of active development:

**Social Evolution Research**:
- **Communication semantics**: Analysis of signal meaning and proto-language emergence
- **Cooperation networks**: Reciprocal altruism patterns and kin selection dynamics  
- **Memory specialization**: Recognition systems and behavioral template evolution
- **Social hierarchies**: Leadership emergence and group coordination patterns

**Technical Improvements**:
- **Performance optimization**: Vectorization of social computations, GPU acceleration
- **Advanced analytics**: Social network analysis, signal clustering, behavioral synchronization
- **Visualization enhancements**: Signal visualization, transfer animations, social network overlays
- **Multi-species dynamics**: Predator-prey systems with different social capabilities

**Analysis and Tools**:
- **Behavioral pattern detection**: Automated discovery of emergent social behaviors
- **Long-term studies**: Multi-million timestep experiments with social evolution tracking
- **Comparative studies**: Social vs asocial populations, different selection pressures
- **Interactive analysis**: Real-time manipulation of social parameters during simulation

Please open an issue to discuss before starting major changes.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{evolving_social_intelligence,
  author = {Joseph Jabbar},
  title = {Evolving Social Intelligence: An Artificial Life Simulation},
  year = {2025},
  url = {https://github.com/jabbarman/evolving-social-intelligence}
}
```

---

## License

[MIT License](LICENSE) - Feel free to use, modify, and distribute.

---

## Acknowledgments

Inspired by:
- Artificial Life research (Tierra, Avida, Polyworld)
- Multi-agent reinforcement learning (OpenAI, DeepMind)
- Evolutionary computation and open-ended evolution
- Karl Sims' evolved virtual creatures

Built with: NumPy, Pygame, Python

---

## Contact

Questions? Suggestions? Found interesting emergent behaviors?

- Open an issue on GitHub
- Email: jabbarman@gmail.com

**Let's explore what emerges when agents evolve together! 🧬🤖**
