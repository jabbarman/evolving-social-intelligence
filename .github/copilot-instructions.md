# Copilot Instructions: Evolving Social Intelligence

This is an artificial life simulation where neural network agents evolve intelligence through social interaction. Agents navigate a 2D grid world, consume food, reproduce with mutation, and develop emergent behaviors over evolutionary time.

## Build, Test, and Lint Commands

### Running the Simulation
```bash
# Basic run with visualization
python3 main.py

# Fast run without visualization (10x faster)
python3 main.py --no-viz

# Run with custom config
python3 main.py --config configs/fast_test.yaml --steps 5000

# Resume from checkpoint
python3 main.py --resume-from experiments/logs/checkpoints/checkpoint_00050000.pkl.gz --no-viz
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_simulation_smoke.py

# Run single test
pytest tests/test_simulation_smoke.py::test_simulation_runs_for_a_few_steps
```

### Development Dependencies
```bash
# Install with development extras
pip install -e ".[dev]"

# Code formatting
black src/ tests/ main.py

# Linting
flake8 src/ tests/ main.py

# Type checking
mypy src/
```

### Quick Smoke Test
```bash
python3 main.py --config configs/fast_test.yaml --steps 100 --no-viz
```

## High-Level Architecture

### Core Components
- **`main.py`**: Entry point with argument parsing and simulation orchestration
- **`src/simulation.py`**: Main simulation coordinator that orchestrates all components
- **`src/agent.py`**: Individual agents with neural network brains and sensors
- **`src/brain.py`**: NumPy-based neural networks (52→32→6 architecture)
- **`src/environment.py`**: 2D toroidal grid world with food spawning
- **`src/evolution.py`**: Asexual reproduction with mutation and selection
- **`src/analysis.py`**: Metrics logging and behavioral analysis
- **`src/lineage.py`**: Ancestry tracking and lineage analytics
- **`src/visualization.py`**: Real-time pygame rendering

### Agent Architecture
Agents have 52 inputs (5×5 perception grid: 25 food + 25 agents + own energy + age) feeding through a 32-neuron hidden layer to 6 outputs (5 movement directions via softmax + 1 communication signal). Each agent carries ~1,700 parameters evolved through mutations.

### Simulation Loop
Each timestep: perception → neural network decision → movement → food consumption → metabolism → reproduction (>150 energy) → death (≤0 energy) → population cap enforcement → food spawning → metrics logging.

### Data Flow
`Simulation` coordinates: `Environment` manages grid state → `Agent` instances perceive and act → `Evolution` handles reproduction/mutation → `Logger` records metrics → `LineageTracker` maintains ancestry → `Visualizer` renders state.

## Key Conventions

### Configuration System
- All parameters defined in YAML configs under `configs/`
- Five main sections: `simulation`, `environment`, `agent`, `brain`, `evolution`
- Optional sections: `logging`, `behavioral_metrics`, `lineage_tracking`
- Config validation happens at runtime during `Simulation` initialization

### Agent ID Management
- Agents use `itertools.count()` for unique IDs across simulation lifetime
- Parent-child relationships tracked via `parent_id` and `generation` fields
- Lineage root ID preserved through reproduction for ancestry tracking

### Spatial Optimization
- Agents stored in `agent_grid` dict for O(1) spatial lookups: `{(x,y): [agent1, agent2]}`
- Toroidal world: coordinates wrap around grid boundaries
- Perception uses relative offsets from agent position with wraparound

### Energy Economics
- Base metabolic cost charged every timestep regardless of action
- Movement incurs additional cost beyond base metabolism
- Reproduction transfers fixed energy amount (80) from parent to offspring
- Death occurs when energy ≤ 0 (checked after metabolism phase)

### Mutation Strategy
- Fixed fraction of weights mutated (10% by default)
- Gaussian noise N(0, 0.1) added to selected weights
- No crossover - purely asexual reproduction
- Mutation parameters configurable via `evolution.mutation_rate` and `evolution.mutation_std`

### Logging and Checkpointing
- Metrics aggregated at configurable intervals and saved to `experiments/logs/metrics.npz`
- Automatic checkpointing during long runs via `checkpoint_interval`
- Ctrl+C triggers best-effort checkpoint save before exit
- Behavioral metrics (movement entropy, foraging efficiency) logged separately if enabled
- Lineage snapshots saved to SQLite database for ancestry analysis

### Testing Patterns
- Smoke tests use `configs/fast_test.yaml` (small 50×50 grid, 20 agents)
- Test functions accept `tmp_path` fixture for isolated logging directories
- Tests verify simulation can run for a few steps without crashing
- Checkpoint tests ensure state roundtrip fidelity including RNG state

### File Organization
- Source code in `src/` with clear module separation
- Configs in `configs/` as YAML files
- Test configs should override `logging.save_dir` to use temporary paths
- Experimental results under `experiments/logs/`
- Analysis notebooks in `notebooks/`
- Plotting scripts in `scripts/`

### Performance Considerations
- Disable visualization (`--no-viz`) for 10x speedup during long runs
- NumPy vectorization used throughout for computational efficiency
- Spatial indexing prevents O(n²) agent-agent distance calculations
- Consider smaller grids or populations for faster iteration during development