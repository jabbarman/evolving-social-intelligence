# Evolving Social Intelligence: Technical Specification v1.0

## Project Vision

Create an open-ended evolutionary environment where AI agents develop intelligence through social interaction, recapitulating the environmental and social pressures that shaped biological intelligence on Earth.

**Core Principle**: Provide agents with latent capabilities that only become useful as environmental and social complexity emerges through their own actions.

**Timeline**: Long-term project (2-5+ years) with meaningful milestones every 3-6 months.

---

## Phase 1: Minimal Viable Environment (MVP)

### 1.1 Environment Specification

**World Structure**:
- 2D grid-based world (initially 100x100 cells, configurable)
- Toroidal topology (edges wrap) to avoid boundary effects
- Discrete time steps
- Each cell can contain: empty space, resources, agents, or structures (future)

**Resource System**:
- Single resource type initially: "food"
- Food spawns probabilistically in empty cells (configurable rate)
- Food persists until consumed
- Each food unit has energy value (default: 10 units)

**Environmental Physics**:
- No obstacles initially (can add later)
- No terrain variation (all cells equivalent)
- Simple, deterministic movement rules

### 1.2 Agent Architecture

**Agent Embodiment**:
```python
class Agent:
    position: (x, y)
    energy: float (starts at 100)
    age: int (timesteps alive)
    genome: neural network weights
    memory: short-term state (optional initially)
```

**Agent Sensors** (local 5x5 grid perception):
- Food locations (binary grid: 25 values)
- Other agents (binary grid: 25 values)  
- Own energy level (1 value)
- Own age (1 value)
- **Total inputs: 52 values**

**Agent Actuators**:
- Movement: 5 actions (up, down, left, right, stay)
- Consume: automatic if on food cell
- Communication: simple signal emission (1 float value, initially unused)

**Agent Neural Network**:
- Input layer: 52 neurons
- Hidden layer: 32 neurons (ReLU activation)
- Output layer: 6 neurons (5 movement + 1 communication signal)
- **Total parameters: ~1,700** (small enough for fast evolution)
- Use softmax for movement selection

**Energy Dynamics**:
- Energy decreases by 1 per timestep (base metabolic cost)
- Movement costs additional 0.1 energy
- Consuming food adds its energy value
- Agent dies when energy ≤ 0

### 1.3 Evolutionary Mechanics

**Reproduction**:
- Asexual reproduction initially
- Trigger: energy > reproduction_threshold (default: 150)
- Cost: 80 energy from parent
- Offspring spawns in adjacent empty cell
- Offspring inherits parent genome with mutations

**Mutation**:
- Gaussian noise added to weights: N(0, σ) where σ = 0.1
- 10% of weights mutated per reproduction
- Mutation rate itself could evolve later

**Selection**:
- Natural: agents die when energy depletes
- Population cap: 200 agents (oldest die if exceeded)
- No explicit fitness function beyond survival/reproduction

**Population Initialization**:
- Start with 50 agents with random neural network weights
- Distributed randomly across grid
- Initial energy: 100 each

### 1.4 Simulation Loop

```python
for timestep in range(max_timesteps):
    # 1. Agent perception
    for agent in agents:
        observations = get_local_observations(agent)
    
    # 2. Agent decision
    for agent in agents:
        actions = agent.network.forward(observations)
    
    # 3. Execute actions
    for agent in agents:
        execute_movement(agent, actions)
        execute_consumption(agent)
        update_energy(agent)
    
    # 4. Reproduction
    for agent in agents:
        if agent.energy > reproduction_threshold:
            create_offspring(agent)
    
    # 5. Death and cleanup
    remove_dead_agents()
    enforce_population_cap()
    
    # 6. Environment update
    spawn_food()
    
    # 7. Logging
    log_metrics(timestep)
```

---

## Phase 2: Latent Capabilities (Built-in but Initially Dormant)

These capabilities exist in the agent architecture from the start but won't be useful until environmental complexity increases:

### 2.1 Communication System

**Signal Emission**:
- Agents emit a continuous signal value [-1, 1]
- Signal visible to agents in local perception range
- No predefined meaning - agents must learn to use/interpret

**Implementation**:
- Add to sensor inputs: signals from nearby agents (8 values max)
- Already in output layer (1 neuron)
- Cost: 0.5 energy to emit non-zero signal

### 2.2 Resource Transfer

**Capability**:
- Agents can transfer energy to adjacent agents
- Requires both agents to "agree" (both output transfer action)
- Amount: 10 energy units

**Implementation**:
- Add action output neuron: transfer (yes/no)
- Matching logic in simulation loop
- Initially unlikely to emerge; becomes useful for kin/reciprocal altruism

### 2.3 Memory System

**Short-term Memory**:
- Internal state vector (16 values)
- Fed back as additional inputs each timestep
- Allows temporal reasoning, recognition

**Implementation**:
- Add recurrent connections or LSTM-like mechanism
- Keep simple initially (just state vector)

### 2.4 Structure Building (Future)

**Placeholder**:
- Agents can place "markers" on grid cells
- Cost energy but create persistent environmental features
- Could become: territorial markers, resource caches, etc.

---

## Phase 3: Technology Stack

### 3.1 Core Components

**Language**: Python 3.10+
- Readable, accessible to contributors
- Rich ecosystem for ML and visualization

**Numerical Computing**:
- NumPy for arrays and grid operations
- PyTorch or JAX for neural networks (PyTorch recommended - more familiar to contributors)

**Visualization**:
- Pygame for real-time visualization during development
- Matplotlib for analysis and plotting
- Optional: Web-based visualization (Plotly/Dash) for remote monitoring

**Configuration**:
- YAML or JSON for all parameters
- Separate config files for: environment, agents, evolution, logging

**Version Control**:
- Git/GitHub
- Clear branching strategy (main, develop, feature branches)

### 3.2 Project Structure

```
evolving-social-intelligence/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml
│   ├── experiments/
├── src/
│   ├── __init__.py
│   ├── environment.py      # World, resources, physics
│   ├── agent.py            # Agent class, sensors, actuators
│   ├── brain.py            # Neural network implementation
│   ├── evolution.py        # Reproduction, mutation, selection
│   ├── simulation.py       # Main simulation loop
│   ├── visualization.py    # Pygame rendering
│   └── analysis.py         # Metrics, logging, analysis
├── experiments/
│   └── logs/               # Experiment results
├── tests/
│   └── test_*.py
└── docs/
    ├── architecture.md
    ├── getting_started.md
    └── contributing.md
```

### 3.3 Performance Considerations

**Optimization Targets**:
- 1000+ timesteps per second on modest hardware
- Support 500+ agents without significant slowdown
- Efficient grid operations (use NumPy vectorization)

**Profiling Points**:
- Agent perception (can batch)
- Neural network forward passes (can parallelize)
- Grid updates

**Future Scaling**:
- GPU acceleration for agent inference
- Distributed simulation across multiple machines
- Checkpoint/resume for long runs

---

## Phase 4: Metrics and Observables

### 4.1 Population Metrics
- Population size over time
- Birth/death rates
- Age distribution
- Energy distribution

### 4.2 Behavioral Metrics
- Movement patterns (spread, clustering)
- Resource consumption efficiency
- Communication signal usage
- Interaction frequency

### 4.3 Evolutionary Metrics
- Genetic diversity (weight variance)
- Lineage tracking (family trees)
- Behavioral diversity (action distributions)
- Selection pressure indicators

### 4.4 Emergence Indicators
- Spatial clustering (coordination?)
- Communication pattern complexity
- Resource transfer events (cooperation?)
- Novel behaviors (unexpected action sequences)

### 4.5 Logging Strategy
- Timestep-level: basic metrics (fast)
- Checkpoint saves: every 10K timesteps (full state)
- Analysis runs: every 50K timesteps (compute expensive metrics)
- Video captures: periodic visualization snapshots

---

## Phase 5: Development Milestones

### Milestone 1: Simulation Core (Weeks 1-4)
**Goal**: Working simulation with random agents

**Deliverables**:
- Environment implementation
- Agent sensing and acting
- Basic visualization
- Configuration system

**Success Criteria**: 
- Simulation runs stably for 100K timesteps
- Agents move and consume resources
- Population dynamics observable

### Milestone 2: Evolution (Weeks 5-8)
**Goal**: Agents evolve better behaviors

**Deliverables**:
- Reproduction system
- Mutation mechanism
- Selection pressure
- Lineage tracking

**Success Criteria**:
- Population maintains itself >1M timesteps
- Observable improvement in resource gathering efficiency
- Behavioral diversity maintained

### Milestone 3: Analysis Tools (Weeks 9-10)
**Goal**: Understand what's happening

**Deliverables**:
- Comprehensive metrics logging
- Visualization tools
- Analysis notebooks
- Performance profiling

**Success Criteria**:
- Can identify interesting behaviors
- Can compare runs quantitatively
- Can generate publication-quality figures

### Milestone 4: Open Source Launch (Weeks 11-12)
**Goal**: Ready for collaborators

**Deliverables**:
- Documentation complete
- Code cleaned and tested
- Vision document published
- GitHub infrastructure set up

**Success Criteria**:
- Someone can clone and run in <30 minutes
- Contributing guidelines clear
- Vision compelling

---

## Phase 6: Experimental Progression

### Experiment 1: Baseline (Months 1-2)
**Setup**: MVP as specified above
**Duration**: 5M timesteps (~1 week compute)
**Goal**: Establish baseline, verify system works

### Experiment 2: Environmental Variation (Months 3-4)
**Modifications**: Vary food spawn rates, add resource clustering
**Goal**: Observe adaptation to different ecological pressures

### Experiment 3: Social Pressure (Months 5-6)
**Modifications**: Benefits to proximity, enable communication
**Goal**: First signs of coordination?

### Experiment 4: Long-term Evolution (Months 7-12)
**Setup**: Best configuration from above
**Duration**: 100M+ timesteps (continuous running)
**Goal**: See what emerges over extended time

---

## Phase 7: Open Source Strategy

### 7.1 Community Building

**Target Contributors**:
- ML researchers interested in emergence
- Evolutionary computation people
- Artificial life community
- Game AI developers
- Student researchers

**Outreach Channels**:
- r/MachineLearning, r/artificial, r/ArtificialLife
- Hacker News
- Twitter/X ML community
- Discord servers (AI, ML, evolutionary computation)
- Blog posts on Medium/personal site

**Engagement Strategy**:
- Good First Issues tagged
- Monthly progress updates
- Respond to issues/PRs within 48h
- Credit contributors prominently
- Consider co-authorship on any publications

### 7.2 Governance

**Decision Making**:
- You as primary maintainer initially
- Collaborative for major architectural changes
- Document design decisions in issues/discussions

**Code Standards**:
- PEP 8 style guide
- Type hints required
- Tests for new features
- Documentation for public APIs

---

## Phase 8: Compute Budget Planning

### £250/month (~$310) allocation:

**Option A: Continuous Small-Scale**
- AWS EC2 t3.xlarge or similar: ~£100/month
- Runs 24/7 for baseline experiments
- Remaining budget for larger compute bursts

**Option B: Intermittent Large-Scale**
- Rent GPU instances for intensive periods
- AWS g4dn.xlarge: ~£0.50/hour = £360 for full month
- Use spot instances (can be 70% cheaper)
- Run 2 weeks/month with GPU, analyze the rest

**Option C: Hybrid**
- Small always-on instance for quick experiments: £50/month
- Batch larger runs monthly: £200/month burst

**Recommendation**: Start with Option A, transition to C as needs clarify.

**Storage**:
- S3 for experiment logs: minimal cost for TBs
- Checkpoint files: a few GB per long run

---

## Phase 9: Success Criteria

### Short-term (6 months):
- [ ] Working open source implementation
- [ ] 5+ external contributors
- [ ] Agents evolve efficient foraging
- [ ] System stable for 10M+ timesteps

### Medium-term (12-18 months):
- [ ] Observable social behaviors (clustering, following)
- [ ] Communication signals develop meaning
- [ ] Research blog post or preprint
- [ ] 20+ GitHub stars, active community

### Long-term (2-5 years):
- [ ] Emergent cooperation or complex strategies
- [ ] Novel behaviors not seen in other systems
- [ ] Peer-reviewed publication
- [ ] Other researchers building on this platform

---

## Phase 10: Risk Mitigation

### Technical Risks:
- **Agents evolve boring solutions**: Build in environmental complexity, try different selection pressures
- **Performance bottlenecks**: Profile early, optimize hot paths, consider parallelization
- **Bugs in evolution**: Extensive testing, validation against simple cases

### Project Risks:
- **Loss of motivation**: Set small milestones, share progress publicly, find collaborators
- **Insufficient compute**: Start small, seek academic partnerships or cloud credits (AWS, Google have research programs)
- **No interesting results**: Document negative results, iterate on design, lower expectations

### Community Risks:
- **No contributors**: Project still valuable for your learning; results may attract people later
- **Toxic contributors**: Clear code of conduct, firm but fair moderation
- **Scope creep**: Stay focused on core vision, defer tangential features

---

## Appendix A: Theoretical Grounding

### Related Work:
- **Artificial Life**: Tierra, Avida, Polyworld
- **Multi-agent RL**: OpenAI hide-and-seek, DeepMind capture-the-flag
- **Evolutionary Robotics**: Karl Sims, evolved virtual creatures
- **Social Evolution**: Axelrod's cooperation tournaments
- **Open-endedness**: POET, Lenia, neural cellular automata

### Key Papers to Review:
- "The Evolution of Cooperation" - Axelrod
- "Open-Ended Evolution" - various authors
- "Emergence of Grounded Compositional Language" - Mordatch & Abbeel
- "Embodied Artificial Intelligence" - Pfeifer & Scheier

### Novelty of This Approach:
- Combines evolutionary algorithms with deep learning
- Explicit recapitulation of evolutionary stages
- Open-ended social complexity as primary goal
- Open source and long-term focused

---

## Appendix B: Configuration Template

```yaml
# default.yaml
simulation:
  grid_size: [100, 100]
  max_timesteps: 1000000
  population_cap: 200
  initial_population: 50
  seed: 42

environment:
  food_spawn_rate: 0.01  # probability per cell per timestep
  food_energy_value: 10

agent:
  initial_energy: 100
  base_metabolic_cost: 1.0
  movement_cost: 0.1
  perception_range: 2  # cells in each direction (5x5 grid)
  reproduction_threshold: 150
  reproduction_cost: 80

brain:
  input_size: 52
  hidden_size: 32
  output_size: 6
  activation: relu

evolution:
  mutation_rate: 0.1  # fraction of weights
  mutation_std: 0.1

logging:
  log_interval: 1000  # timesteps
  checkpoint_interval: 10000
  analysis_interval: 50000
  visualization_fps: 30
```

---

## Next Steps

1. **Set up development environment** (Day 1)
   - Create GitHub repo
   - Initialize Python project structure
   - Set up virtual environment

2. **Implement environment** (Week 1)
   - Grid world
   - Food spawning
   - Basic visualization

3. **Implement agents** (Week 2)
   - Agent class
   - Neural network
   - Sensing and acting

4. **Implement evolution** (Week 3)
   - Reproduction
   - Mutation
   - Selection

5. **Testing and refinement** (Week 4)
   - Run first experiments
   - Fix bugs
   - Tune parameters

6. **Prepare for open source** (Weeks 5-6)
   - Write documentation
   - Create examples
   - Polish code

---

**Good luck! This is going to be fascinating. Feel free to reach out as you build - I'm excited to see where this goes.**