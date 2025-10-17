# Contributing to Evolving Social Intelligence

Thank you for your interest in contributing! This project welcomes contributions from researchers, developers, and anyone interested in artificial life and evolutionary systems.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version)
- Relevant configuration files or logs

### Suggesting Features

Feature suggestions are welcome! Please open an issue describing:
- The feature and its motivation
- How it fits with the project's goals
- Potential implementation approach (if you have ideas)

### Contributing Code

1. **Fork the repository** and create a branch for your feature/fix
2. **Write clean, documented code** following the existing style
3. **Test your changes** - make sure the simulation still runs
4. **Update documentation** if needed (README, docstrings)
5. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/evolving-social-intelligence.git
cd evolving-social-intelligence

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python3 main.py --config configs/fast_test.yaml --steps 100 --no-viz
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use type hints for function parameters and returns
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- Add comments for complex logic

Example:
```python
def calculate_energy(agent: Agent, config: Dict[str, Any]) -> float:
    """Calculate energy cost for an agent's actions.

    Args:
        agent: The agent to calculate costs for
        config: Configuration dictionary with cost parameters

    Returns:
        Total energy cost as a float
    """
    base_cost = config["base_metabolic_cost"]
    movement_cost = config["movement_cost"] if agent.moved else 0
    return base_cost + movement_cost
```

## Areas for Contribution

### High Priority
- **Performance optimization**: Vectorize operations, profile bottlenecks
- **Visualization improvements**: Better graphics, plotting tools
- **Analysis tools**: Behavioral metrics, lineage tracking
- **Documentation**: Tutorials, examples, architecture guides

### Medium Priority
- **Unit tests**: Test coverage for core modules
- **Configuration validation**: Check configs before running
- **Checkpointing**: Save/resume long simulations
- **Experiment tracking**: Better organization of results

### Research Directions
- **Environmental complexity**: Obstacles, terrain, multiple resources
- **Social mechanisms**: Advanced communication, cooperation
- **Predator-prey dynamics**: Multiple agent types
- **Open-ended evolution**: Novelty search, quality diversity

## Testing Your Changes

Before submitting a PR:

1. **Run a quick test**:
```bash
python3 main.py --config configs/fast_test.yaml --steps 500 --no-viz
```

2. **Check that population stabilizes** (doesn't immediately go extinct)

3. **Test with visualization**:
```bash
python3 main.py --config configs/fast_test.yaml --steps 500
```

4. **Verify metrics are saved** and look reasonable

5. **Run on both default and fast_test configs** if you changed core logic

## Pull Request Process

1. Update README.md if you added features or changed usage
2. Update docstrings and inline comments
3. Ensure your PR description explains:
   - What you changed
   - Why you changed it
   - How to test it
4. Link related issues if applicable
5. Be responsive to feedback and questions

## Community Guidelines

- Be respectful and constructive
- Focus on ideas, not individuals
- Welcome newcomers and help them learn
- Share interesting results and discoveries
- Credit others' contributions

## Questions?

- Open a discussion issue for design questions
- Check existing issues for similar topics
- Tag maintainers if you need specific expertise

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for helping make this project better!** 🧬🤖
