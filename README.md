<!-- filepath: c:\Users\u0122906\Documents\pytrebuchet\README.md -->
# Pytrebuchet

A Python-based trebuchet simulation project.

## Description

Pytrebuchet is a physics simulation tool for modeling and analyzing trebuchet mechanics. It supports multiple trebuchet types including:
- **Hinged counterweight trebuchets** - traditional design with a pivoting counterweight
- **Whipper trebuchets** - a whipper trebuchet features a hinged counterweight system, but with the counterweight hanger positioned at the top of the throwing arm

The differential equations governing a hinged counterweight trebuchet's motion were derived using Langrangian mechanics by E. Constans and A. Constans. They published their equations on [http://www.benchtophybrid.com/TB/TB_Literature.html](http://www.benchtophybrid.com/TB/TB_Literature.html). The code uses scipy's solve_ivp function to solve the system of ODEs.

The prediction of the projectile's trajectory by pytrebuchet of a hinged counterweight trebuchet has been verified by comparing it to the prediction made by [https://virtualtrebuchet.com/](https://virtualtrebuchet.com/) for the same input parameters.

The whipper-style trebuchet uses the same differential equations as the hinged counterweight trebuchet, but with additional constraints equations that constrain the motion of the weight and projectile sling segments at the start of the launch.

## Installation

### From source

1. Clone the repository:
```bash
git clone https://github.com/benvb-97/pytrebuchet.git
cd pytrebuchet
```

2. Install the package:
```bash
pip install -e .
```

### Requirements

- Python 3.12 or higher
- numpy >= 2.3.4
- scipy >= 1.16.3
- matplotlib >= 3.10.7
- sympy >= 1.14.0 (for the jupyter notebooks)

## Quick Start

```python
from pytrebuchet import Trebuchet, Projectile, Simulation

# Create a hinged counterweight trebuchet with default parameters
trebuchet = Trebuchet.default_hcw()
projectile = Projectile.default()

# Create and solve a trebuchet simulation
simulation = Simulation(trebuchet, projectile)
simulation.solve()

# Get results
print(f"Horizontal range: {simulation.distance_traveled:.2f} m")
```

## Examples

Check out the [`examples/`](examples/) directory for Jupyter notebooks demonstrating (among others):
- [`plot_initial_position.ipynb`](examples/plot_initial_position.ipynb) - Visualizing the initial trebuchet configuration
- [`whipper_simulation.ipynb`](examples/whipper_simulation.ipynb) - Simulating and animating a whipper trebuchet

## Running Tests

Run the test suite using pytest:

```bash
pytest tests/
```

## Project Structure

```
pytrebuchet/
├── src/pytrebuchet/      # Main package code
├── tests/                # Test suite
├── examples/             # Example notebooks
├── pyproject.toml        # Project configuration
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

- E. Constans and A. Constans, "The Trebuchet: A Gravity-Operated Siege Engine" - [http://www.benchtophybrid.com/TB/TB_Literature.html](http://www.benchtophybrid.com/TB/TB_Literature.html)
- Virtual Trebuchet - [https://virtualtrebuchet.com/](https://virtualtrebuchet.com/)

## License

Pytrebuchet is provided under an MIT license that can be found in the LICENSE file.
