[![Linter](https://github.com/benvb-97/pytrebuchet/actions/workflows/linter.yml/badge.svg)](https://github.com/benvb-97/pytrebuchet/actions/workflows/linter.yml)
[![Run Tests](https://github.com/benvb-97/pytrebuchet/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/benvb-97/pytrebuchet/actions/workflows/tests.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9044d19fa45840ea982c662294695d9a)](https://app.codacy.com/gh/benvb-97/pytrebuchet/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![codecov](https://codecov.io/gh/benvb-97/pytrebuchet/graph/badge.svg?token=D4L0786S9N)](https://codecov.io/gh/benvb-97/pytrebuchet)

# Pytrebuchet

A Python-based trebuchet simulation project.

Full documentation is available at [https://benvb-97.github.io/pytrebuchet/](https://benvb-97.github.io/pytrebuchet/)

![Trebuchet Launch Animation](docs/source/figures/trebuchet_launch.gif)

## Description

[Pytrebuchet](https://benvb-97.github.io/pytrebuchet/) is a physics simulation tool for modeling and analyzing trebuchet mechanics. It supports multiple trebuchet types including:
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

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for a detailed guide.

## References

- E. Constans and A. Constans, "The Trebuchet: A Gravity-Operated Siege Engine" - [http://www.benchtophybrid.com/TB/TB_Literature.html](http://www.benchtophybrid.com/TB/TB_Literature.html)
- Virtual Trebuchet - [https://virtualtrebuchet.com/](https://virtualtrebuchet.com/)

## License

Pytrebuchet is provided under an MIT license that can be found in the [LICENSE](LICENSE) file.
