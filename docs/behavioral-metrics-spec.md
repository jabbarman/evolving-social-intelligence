# Requirements: Behavioral Metrics & Lineage Tracking

## Overview
Add comprehensive behavioral analysis and lineage tracking to understand how agents evolve over time and identify successful genetic lines.

---

## 1. Behavioral Metrics

### 1.1 Average Distance Moved Per Step

**Purpose:** Detect if agents evolve directed movement vs. random wandering

**Implementation:**
- Track each agent's position change per timestep
- Calculate Euclidean distance: `sqrt((x_new - x_old)² + (y_new - y_old)²)`
- Aggregate across population each logging interval

**Metrics to log:**
- `mean_distance_per_step`: Average across all living agents
- `std_distance_per_step`: Standard deviation (shows behavioral diversity)
- `median_distance_per_step`: Median movement distance

**Data structure:**
```python
# Add to metrics.json
{
  "timesteps": [...],
  "mean_distance_per_step": [0.85, 0.87, 0.91, ...],
  "std_distance_per_step": [0.12, 0.15, 0.13, ...],
  "median_distance_per_step": [0.82, 0.85, 0.89, ...]
}
```

**Expected patterns:**
- Random movement: ~0.8-1.0 (agents move in various directions)
- Directed search: >1.2 (agents move further, more purposefully)
- Minimal movement: <0.5 (agents stay put, wait for food)

---

### 1.2 Food Discovery Rate

**Purpose:** Measure foraging efficiency - are agents getting better at finding food?

**Implementation:**
- Track when agents move onto a cell with food
- Calculate: `food_discoveries / total_moves` per agent
- Aggregate across population

**Metrics to log:**
- `mean_food_discovery_rate`: Average rate across population
- `max_food_discovery_rate`: Best performer's rate
- `total_food_consumed`: Total food eaten this interval

**Data structure:**
```python
# Add to metrics.json
{
  "timesteps": [...],
  "mean_food_discovery_rate": [0.02, 0.025, 0.031, ...],
  "max_food_discovery_rate": [0.05, 0.06, 0.08, ...],
  "total_food_consumed": [15, 18, 22, ...]
}
```

**Per-agent tracking:**
```python
# Add to Agent class
self.food_discoveries = 0
self.total_moves = 0
self.discovery_rate = 0.0  # Updated each timestep
```

**Expected patterns:**
- Improving over time: discovery rate increases
- Plateau: rate stabilizes at environment's theoretical maximum
- Decline: overpopulation reduces food availability

---

### 1.3 Movement Entropy

**Purpose:** Detect if movement becomes more predictable (patterns) vs. random

**Implementation:**
- Track last N moves per agent (e.g., N=20)
- Calculate Shannon entropy of direction distribution
- High entropy = random/exploratory, Low entropy = patterned/systematic

**Formula:**
```
H = -Σ(p_i * log₂(p_i))

where p_i = probability of direction i (up/down/left/right/stay)
```

**Metrics to log:**
- `mean_movement_entropy`: Average across population
- `min_movement_entropy`: Most patterned agent
- `max_movement_entropy`: Most random agent

**Data structure:**
```python
# Add to metrics.json
{
  "timesteps": [...],
  "mean_movement_entropy": [2.1, 2.0, 1.8, ...],  # Max = 2.32 bits for 5 actions
  "min_movement_entropy": [1.2, 1.0, 0.8, ...],
  "max_movement_entropy": [2.3, 2.3, 2.2, ...]
}
```

**Per-agent tracking:**
```python
# Add to Agent class
self.recent_actions = deque(maxlen=20)  # Last 20 moves
self.movement_entropy = 0.0  # Calculated periodically
```

**Expected patterns:**
- Random agents: H ≈ 2.3 bits (uniform distribution)
- Patterned movement: H < 1.5 bits (e.g., mostly moving in one direction)
- Mixed strategies: H ≈ 1.8-2.0 bits

**Interpretation:**
- Decreasing entropy over time = agents evolving systematic search patterns
- Sustained high entropy = exploration remains beneficial
- Bimodal distribution = multiple strategies coexist

---

## 2. Lineage Tracking

### 2.1 Family Tree Structure

**Purpose:** Track genetic ancestry to identify most successful lineages

**Implementation:**
- Assign unique ID to each agent at birth
- Track parent ID when reproducing
- Build family tree in memory/database

**Data structure:**
```python
# Add to Agent class
class Agent:
    def __init__(self, agent_id, parent_id=None):
        self.id = agent_id  # Unique identifier
        self.parent_id = parent_id  # None for founding population
        self.generation = 0 if parent_id is None else parent.generation + 1
        self.offspring_count = 0  # Direct children
        self.lineage = []  # Path to founding ancestor
```

**Lineage tracking file:**
```json
// lineages.json
{
  "agents": {
    "0": {"parent": null, "generation": 0, "offspring": [12, 45, 78]},
    "12": {"parent": 0, "generation": 1, "offspring": [156, 203]},
    "45": {"parent": 0, "generation": 1, "offspring": [189, 234, 267]},
    ...
  },
  "founding_population": [0, 1, 2, 3, ..., 49],
  "current_population": [12034, 12045, 12056, ...]
}
```

---

### 2.2 Descendant Counting

**Purpose:** Which founding agents have the most descendants alive now?

**Metrics to track:**
- `descendants_per_founder`: Map of founding_agent_id → descendant_count
- `dominant_lineages`: Top 5 founding agents by descendants
- `lineage_diversity`: How many founding lineages still exist?
- `extinction_events`: Founding lineages that died out

**Implementation:**
```python
def count_descendants(agent_id, lineage_data):
    """Recursively count all living descendants of agent_id"""
    count = 0
    for current_agent in living_agents:
        if traces_back_to(current_agent, agent_id):
            count += 1
    return count

def get_lineage_stats():
    """Generate lineage statistics for current population"""
    stats = {}
    for founder_id in founding_population:
        stats[founder_id] = {
            "descendants": count_descendants(founder_id),
            "generations": max_generation_depth(founder_id),
            "extinct": descendants == 0
        }
    return stats
```

**Logging frequency:**
- Every 10,000 timesteps (lineage analysis is expensive)
- Save to separate file: `lineage_stats.json`

---

### 2.3 Lineage Metrics

**Data to log:**

```json
// lineage_stats.json
{
  "timestep": 100000,
  "total_agents": 200,
  "active_lineages": 23,  // Founding agents with living descendants
  "extinct_lineages": 27,  // Founding agents with no descendants
  "dominant_lineages": [
    {"founder_id": 7, "descendants": 45, "percentage": 22.5},
    {"founder_id": 23, "descendants": 38, "percentage": 19.0},
    {"founder_id": 41, "descendants": 29, "percentage": 14.5},
    {"founder_id": 12, "descendants": 24, "percentage": 12.0},
    {"founder_id": 3, "descendants": 18, "percentage": 9.0}
  ],
  "lineage_diversity_index": 0.78,  // Simpson's diversity index
  "mean_generation": 156.4,  // Average generation number
  "max_generation": 234  // Deepest generation
}
```

---

### 2.4 Genetic Similarity Tracking (Optional Enhancement)

**Purpose:** Track how genetically similar descendants are to founders

**Implementation:**
- Calculate neural network weight distance between ancestor and descendant
- Track genetic drift over generations

```python
def genetic_distance(agent1, agent2):
    """Calculate L2 distance between neural network weights"""
    weights1 = agent1.brain.get_weights()
    weights2 = agent2.brain.get_weights()
    return np.linalg.norm(weights1 - weights2)
```

**Metric:**
- `mean_genetic_drift`: Average distance from founding ancestors
- Shows how much mutation has accumulated

---

## 3. Implementation Plan

### Phase 1: Basic Behavioral Metrics (2-3 hours)
1. Add movement tracking to Agent class
2. Implement distance, discovery rate, entropy calculations
3. Add to existing logging system
4. Test with short run (10K timesteps)

### Phase 2: Lineage Foundation (2-3 hours)
1. Add ID and parent_id to Agent class
2. Implement lineage data structure
3. Add descendant counting function
4. Save lineage data periodically

### Phase 3: Analysis Tools (2-3 hours)
1. Create visualization functions for behavioral metrics
2. Build family tree visualization
3. Generate lineage reports
4. Add to analysis.py

### Phase 4: Integration & Testing (1-2 hours)
1. Run full 1M timestep experiment with new metrics
2. Verify performance impact (should be minimal)
3. Document new metrics in README
4. Create example analysis notebook

---

## 4. Configuration Options

Add to `config.yaml`:

```yaml
behavioral_metrics:
  enabled: true
  movement_history_length: 20  # For entropy calculation
  log_interval: 100  # Log every N timesteps

lineage_tracking:
  enabled: true
  save_interval: 10000  # Save lineage data every N timesteps
  track_genetic_distance: false  # Optional, more expensive
  max_lineage_depth: 1000  # Prevent infinite recursion
```

---

## 5. Expected Outcomes

### Success Indicators:
- ✅ Movement distance increases over time (directed search evolves)
- ✅ Food discovery rate improves (better foraging)
- ✅ Movement entropy decreases (patterned behavior emerges)
- ✅ Few dominant lineages emerge (successful strategies replicate)
- ✅ Some founding lineages go extinct (selection pressure working)

### Analysis Questions Enabled:
1. Are agents evolving systematic search patterns? (entropy)
2. Is foraging efficiency improving? (discovery rate)
3. Which genetic lines are most successful? (lineage dominance)
4. How quickly do beneficial mutations spread? (lineage growth rate)
5. What movement strategies correlate with success? (distance + descendants)

---

## 6. Performance Considerations

**Computational cost:**
- Movement tracking: ~5% overhead (simple calculations)
- Entropy calculation: ~10% overhead (deque operations)
- Lineage tracking: ~5% overhead (ID management)
- Descendant counting: ~20% overhead (recursive, done infrequently)

**Mitigation:**
- Calculate expensive metrics less frequently
- Use efficient data structures (deque, dict)
- Only count descendants every 10K timesteps
- Option to disable tracking in config

**Storage impact:**
- Behavioral metrics: +30% to metrics.json size
- Lineage data: Separate file, ~1-2MB per 1M timesteps
- Total: Manageable for long runs

---

## 7. File Structure

```
evolving-social-intelligence/
├── src/
│   ├── metrics.py           # Enhanced with behavioral metrics
│   ├── lineage.py           # NEW: Lineage tracking logic
│   └── analysis.py          # Updated with new visualizations
├── experiments/
│   └── logs/
│       ├── metrics.json        # Existing + behavioral metrics
│       ├── lineage_stats.json  # Summary lineage statistics
│       └── lineage.db          # SQLite lineage datastore
└── notebooks/
    └── behavioral_analysis.ipynb  # NEW: Analysis examples
```

---

## 8. Testing Checklist

Before merging:
- [ ] Run 10K timestep test, verify metrics log correctly
- [ ] Confirm performance impact <25% (measure timesteps/second)
- [ ] Verify lineage IDs are unique and consistent
- [ ] Check descendant counting is accurate (manual verification)
- [ ] Test with population cap enforcement (oldest agents removed)
- [ ] Verify metrics.json is still valid JSON
- [ ] Update README with new metrics documentation
- [ ] Create example visualization notebook

---

## 9. Documentation Updates

### README additions:
```markdown
## Behavioral Metrics

The simulation now tracks:
- **Movement patterns**: Distance traveled, entropy of direction choices
- **Foraging efficiency**: Food discovery rate over time
- **Lineage tracking**: Which founding agents' genes persist

See `notebooks/behavioral_analysis.ipynb` for visualization examples.
```

### Config documentation:
Document all new configuration options with examples and expected values.

---

## 10. Future Enhancements

Once basic tracking is working:
- [ ] Spatial heatmaps (where do successful agents forage?)
- [ ] Temporal patterns (do agents develop daily/cyclical behaviors?)
- [ ] Social metrics (when Phase 3 adds interaction)
- [ ] Phylogenetic trees (visualize evolutionary relationships)
- [ ] Mutation tracking (which mutations led to successful lineages?)

---

## Acceptance Criteria

✅ Behavioral metrics log without errors
✅ Performance impact <25%
✅ Lineage tracking accurately identifies descendants
✅ Visualization notebook demonstrates all metrics
✅ Documentation is complete and clear
✅ Config options work as expected
✅ No breaking changes to existing functionality
