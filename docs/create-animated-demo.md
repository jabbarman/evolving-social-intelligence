# Creating Animated Demo for Social Evolution

## Recording the GIF

Since I can't directly create screen recordings, here's how to create the animated demo:

### Option 1: Screen Recording Tools (Recommended)

**On macOS:**
```bash
# Using built-in screen recording
# 1. Start the simulation
python3 main.py --config configs/social_evolution.yaml --steps 1000

# 2. Use QuickTime Player or Screenshot utility to record
# 3. Convert to optimized GIF using ffmpeg or online tools
```

**On Linux:**
```bash
# Using ffmpeg to record X11 display
ffmpeg -f x11grab -s 800x600 -i :0.0+100,100 -t 30 social_demo.mp4

# Convert to optimized GIF
ffmpeg -i social_demo.mp4 -vf "fps=10,scale=600:-1:flags=lanczos,palettegen" palette.png
ffmpeg -i social_demo.mp4 -i palette.png -filter_complex "fps=10,scale=600:-1:flags=lanczos[x];[x][1:v]paletteuse" social-evolution-demo.gif
```

**On Windows:**
```bash
# Use OBS Studio or similar screen recording software
# Then convert to GIF using online tools or ffmpeg
```

### Option 2: Programmatic GIF Creation

```python
# Add to main.py or create separate script
import imageio
import pygame

def create_demo_gif():
    # Record frames during simulation
    frames = []
    sim = Simulation(config)
    viz = Visualizer(grid_size)
    
    for step in range(200):
        sim.step()
        # Capture pygame surface as image
        frame = pygame.surfarray.array3d(viz.screen)
        frames.append(frame)
        
        if step % 10 == 0:  # Every 10th frame
            viz.render(sim.environment, sim.agents, sim.timestep)
    
    # Save as GIF
    imageio.mimsave('docs/social-evolution-demo.gif', frames, fps=5)
```

### GIF Specifications for GitHub README

- **Size**: Max 10MB for smooth loading
- **Dimensions**: 600-800px width recommended  
- **Duration**: 15-30 seconds optimal
- **Frame rate**: 5-10 FPS for smaller file size
- **Content**: Show social clustering, metrics updating, population growth

### What to Capture

1. **Initial random distribution** of agents
2. **Clustering behavior** developing over time
3. **Social metrics** updating in real-time:
   - Communication rate increasing
   - Transfer events occurring
   - Proximity bonuses accumulating
4. **Population growth** to cap with social behaviors
5. **Final state** showing established social patterns

### Filename and Placement

Save as: `docs/social-evolution-demo.gif`
Reference in README: `![Social Behavior Demo](docs/social-evolution-demo.gif)`

## Alternative: Multiple Screenshots

If GIF creation is not feasible, create a series of screenshots:
- `docs/demo-step-000.png` - Initial state
- `docs/demo-step-100.png` - Early clustering  
- `docs/demo-step-500.png` - Social behaviors emerging
- `docs/demo-step-1000.png` - Established social groups

Then use GitHub's image grid syntax:
```markdown
| Initial State | Clustering Emerges | Social Behaviors | Final Groups |
|---------------|-------------------|------------------|--------------|
| ![](docs/demo-step-000.png) | ![](docs/demo-step-100.png) | ![](docs/demo-step-500.png) | ![](docs/demo-step-1000.png) |
```