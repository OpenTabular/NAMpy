# Contributing to NAMpy

Thank you for your interest in contributing to NAMpy! We welcome contributions from the community and are grateful for your support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/NAMpy.git
   cd NAMpy
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/OpenTabular/NAMpy.git
   ```

## Development Setup

### Prerequisites

- Python 3.6 or higher
- pip
- virtualenv (recommended)

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Verify the installation:
   ```bash
   pytest tests/
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **New features**: Add new models or functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Create tutorials or example notebooks
- **Code quality**: Refactoring, optimization, or cleanup

### Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our code style guidelines

3. **Add tests** for any new functionality

4. **Run the test suite** to ensure everything passes:
   ```bash
   pytest tests/
   ```

5. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Code Style

We follow PEP 8 and use automated tools to maintain code quality.

### Formatting

We use **Black** for code formatting:

```bash
black nampy/ tests/
```

### Import Sorting

We use **isort** for organizing imports:

```bash
isort nampy/ tests/
```

### Linting

We use **Ruff** for linting:

```bash
ruff check nampy/ tests/
```

### Type Checking (Optional)

For type hints, we use **mypy**:

```bash
mypy nampy/
```

### Pre-commit Checks

Before committing, run all quality checks:

```bash
# Format code
black nampy/ tests/
isort nampy/ tests/

# Check for issues
ruff check nampy/ tests/

# Run tests
pytest tests/
```

## Testing

### Running Tests

Run the full test suite:

```bash
pytest tests/
```

Run specific test files:

```bash
pytest tests/test_models_regression.py
```

Run with coverage:

```bash
pytest --cov=nampy --cov-report=html tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use descriptive names that explain what is being tested
- Include docstrings for complex tests
- Aim for high code coverage

Example test structure:

```python
def test_nam_regressor_fit():
    """Test that NAMRegressor can fit on sample data."""
    X, y = make_regression(n_samples=100, n_features=5)
    model = NAMRegressor()
    model.fit(X, y, max_epochs=10)
    assert model is not None
```

## Documentation

### Docstring Style

We use **NumPy-style** docstrings:

```python
def function_name(param1, param2):
    """
    Brief description of the function.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    type
        Description of return value.

    Examples
    --------
    >>> function_name(1, 2)
    3
    """
```

### Updating Documentation

- Update docstrings when modifying functions or classes
- Update README.md for user-facing changes
- Update CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/) format


## Reporting Bugs

### Before Submitting a Bug Report

- Check the [issue tracker](https://github.com/OpenTabular/NAMpy/issues) for existing reports
- Try the latest version from the main branch
- Collect information about your environment

### Submitting a Bug Report

Create an issue with:

- **Clear title** describing the problem
- **Steps to reproduce** the bug
- **Expected behavior** vs actual behavior
- **Environment details**:
  - NAMpy version
  - Python version
  - Operating system
  - Relevant dependencies
- **Code snippet** to reproduce (if applicable)
- **Error messages** or stack traces

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature has already been requested
2. Open an issue with:
   - Clear description of the feature
   - Use cases and motivation
   - Possible implementation approach (optional)
   - Willingness to contribute the implementation

## Implementing New Models

NAMpy is designed to be extensible. To add a new model:

1. Create the base model in `nampy/basemodels/`
2. Create the sklearn wrapper(s) in `nampy/models/`
3. Add configuration in `nampy/configs/`
4. Add tests in `tests/`
5. Update documentation

See the README section "Implement Your Own Model" for details.

---

Thank you for contributing to NAMpy! 🎉
