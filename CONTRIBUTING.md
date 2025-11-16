# Contributing to PyTrebuchet

First off, thank you for considering contributing to PyTrebuchet! It's people like you that make PyTrebuchet a great tool for the community.

## Why These Guidelines?

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## What We're Looking For

PyTrebuchet is an open source project and we love to receive contributions! There are many ways to contribute:

- **Writing tutorials or blog posts** about trebuchet physics and simulation
- **Improving the documentation** (docstrings, README, examples)
- **Submitting bug reports and feature requests** via GitHub issues
- **Writing code** that can be incorporated into PyTrebuchet itself
- **Creating example notebooks** demonstrating interesting trebuchet configurations
- **Improving test coverage** to make the codebase more robust
- **Validating simulation results** against physical experiments or other simulators

## What We're NOT Looking For

Please don't use the GitHub issue tracker for support questions. For general Python or scientific computing questions, Stack Overflow is a better venue. For questions specific to trebuchet physics or this library's usage, feel free to open a discussion or create a well-scoped issue.

# Ground Rules

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior by opening an issue or contacting the project maintainer.

## Technical Responsibilities

- **Ensure cross-platform compatibility** for every change that's accepted (Windows, macOS, Linux)
- **Write tests** for new features and bug fixes. Run `pytest tests/` to verify all tests pass
- **Maintain test coverage**. Run `pytest tests/ --cov=pytrebuchet` to check coverage
- **Follow Python best practices**. Code is formatted with Ruff (`ruff format`)
- **Create issues for major changes**. Discuss things transparently and get community feedback before implementing large features
- **Keep pull requests focused**. One feature or bug fix per PR makes review easier
- **Update documentation**. If you change APIs or add features, update docstrings and examples
- **Write clear commit messages**. Explain what and why, not just how

# Your First Contribution

Unsure where to begin contributing to PyTrebuchet? You can start by looking through issues.

**New to open source?** Here are some friendly tutorials:
- [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)
- [First Timers Only](http://www.firsttimersonly.com/)
- [Make a Pull Request](http://makeapullrequest.com/)

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first 😺

# Getting Started

## Setting Up Your Development Environment

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/benvb-97/pytrebuchet.git
   cd pytrebuchet
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e .[dev]
   ```

4. **Create a branch for your changes**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Running Tests

Before submitting your changes, make sure all tests pass:

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src/pytrebuchet --cov-report=term

# Run specific test file
pytest tests/test_simulation.py

# Run specific test function
pytest tests/test_simulation.py::test_solve
```

## Code Quality

We use Ruff for code formatting and linting:

```bash
# Format code
ruff format .

# Check for linting issues
ruff check .

# Fix auto-fixable linting issues
ruff check --fix .
```

## Making Changes

### For Small or "Obvious" Fixes

Small contributions such as fixing spelling errors, where the content is small enough to not be considered intellectual property, can be submitted directly as a pull request without creating an issue first.

Examples of small fixes:
- Spelling or grammar fixes
- Typo corrections, whitespace and formatting changes
- Comment cleanup
- Bug fixes that change default return values or error codes stored in constants
- Adding logging messages or debugging output
- Changes to metadata files like `.gitignore`, build scripts, etc.

### For Larger Changes

1. **Create an issue first** to discuss the change
2. **Wait for feedback** from maintainers before starting work
3. **Fork the repository** and create a branch
4. **Make your changes** with appropriate tests
5. **Run the test suite** to ensure nothing broke
6. **Submit a pull request** referencing the issue

# How to Report a Bug

## Security Vulnerabilities

If you find a security vulnerability, **do NOT open an issue**. Contact the project maintainer directly at the email listed on [the GitHub profile](https://github.com/benvb-97).

## Bug Reports

When filing a bug report, please include:

1. **PyTrebuchet version** (`python -c "import pytrebuchet; print(pytrebuchet.__version__)"`)
2. **Python version** (`python --version`)
3. **Operating system and version**
4. **What you did** (minimal code example to reproduce)
5. **What you expected to see**
6. **What you saw instead** (error messages, incorrect output, etc.)

**Example bug report:**

```markdown
## Bug Description
Simulation fails when sling length is zero

## Steps to Reproduce
```python
from pytrebuchet import Trebuchet, Projectile, Simulation

trebuchet = Trebuchet.default()
trebuchet.l_sling_projectile = 0.0  # This causes the issue
projectile = Projectile.default()
simulation = Simulation(trebuchet, projectile)
simulation.solve()  # Raises ZeroDivisionError
```

## Expected Behavior
Should raise a descriptive ValueError about invalid sling length

## Actual Behavior
Raises ZeroDivisionError with cryptic message

## Environment
- Python: 3.12.0
- OS: Windows 11
```

# How to Suggest a Feature

## Project Philosophy

PyTrebuchet aims to be:

- **Physically accurate**: Simulations should reflect real trebuchet physics
- **Well-documented**: Code should be clear with comprehensive docstrings
- **Easy to use**: Simple default configurations for common use cases
- **Extensible**: Support for multiple trebuchet types and custom configurations
- **Scientifically rigorous**: Validated against established models and data

## Feature Requests

If you find yourself wishing for a feature that doesn't exist in PyTrebuchet, you are probably not alone! Open an issue that describes:

1. **The feature you would like to see**
2. **Why you need it** (use case)
3. **How it should work** (API design if applicable)
4. **Whether you're willing to implement it** (we can guide you!)

# Code Review Process

The core maintainer reviews pull requests regularly. Here's what to expect:

1. **Initial review** within 1-2 weeks of submission
2. **Feedback and discussion** on the PR if changes are needed
3. **Approval and merge** once all feedback is addressed and tests pass
4. **Response expectation**: If we request changes, please respond within two weeks. After two weeks of inactivity, we may close the PR.

## What We Look For

- Does the PR address a real need?
- Are tests included and passing?
- Is the code well-documented?
- Does it follow the existing code style?
- Is the PR focused on one feature/fix?

# Commit Message Conventions

We follow a simple commit message format:

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring without behavior change
- `chore`: Maintenance tasks

**Example:**
```
feat: add whipper trebuchet support

Implemented differential equations for whipper-style trebuchets
with constrained phases. Added WhipperTrebuchet class and updated
simulation to handle multiple phase transitions.

Closes #42
```

# Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers!

Thank you for contributing to PyTrebuchet! 🎯
