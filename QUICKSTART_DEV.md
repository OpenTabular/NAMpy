# NAMpy Developer Quick Start

## Initial Setup

```bash
# Clone and enter directory
git clone https://github.com/OpenTabular/NAMpy.git
cd NAMpy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
make install-dev

# Or manually:
pip install -e ".[dev]"
pip install pre-commit
pre-commit install
```

## Common Commands

### Development
```bash
make format        # Format code (black + isort)
make lint          # Check code quality (ruff)
make test          # Run tests
make test-cov      # Run tests with coverage
make type-check    # Type checking (mypy)
make quality       # Run all checks
```

### Documentation
```bash
make docs          # Build HTML documentation
make docs-serve    # Build and serve on localhost:8000
make docs-clean    # Clean documentation build
```

### Building
```bash
make clean         # Clean build artifacts
make build         # Build distribution packages
```

### Publishing
```bash
make publish-test  # Publish to TestPyPI
make publish       # Publish to PyPI (use with caution!)
```

## Quick Testing

```python
# Test basic import
python -c "import nampy; print(nampy.__version__)"

# Test model import
python -c "from nampy.models import NAMRegressor; print('OK')"

# Quick regression test
python << EOF
from nampy.models import NAMRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=5, random_state=42)
model = NAMRegressor()
model.fit(X, y, max_epochs=10)
print(f"Score: {model.score(X, y):.4f}")
EOF
```

## Pre-Commit Hooks

Automatically run on `git commit`:
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON/TOML validation
- Black formatting
- isort import sorting
- Ruff linting
- mypy type checking (optional)

Run manually:
```bash
pre-commit run --all-files
```

## File Structure

```
NAMpy/
├── nampy/                 # Main package
│   ├── __init__.py       # Public API
│   ├── __version__.py    # Version info
│   ├── models/           # Model implementations
│   ├── neural/modules/   # PyTorch architectures
│   ├── neural/configs/   # Configuration dataclasses
│   ├── preprocessing/    # Data preprocessing
│   ├── utils/            # Utilities
│   └── ...
├── tests/                # Test suite
├── .github/workflows/    # CI/CD pipelines
├── README.md            # Main documentation
├── CONTRIBUTING.md      # Contribution guide
├── CHANGELOG.md         # Version history
├── LICENSE              # MIT License
├── pyproject.toml       # Modern packaging config
├── setup.py             # Legacy packaging (compatibility)
├── Makefile             # Development commands
└── requirements.txt     # Dependencies
```

## Workflow

### Adding a Feature
```bash
# Create branch
git checkout -b feature/my-feature

# Make changes
# ... edit files ...

# Format and test
make format
make quality

# Commit
git add .
git commit -m "Add feature: description"

# Push
git push origin feature/my-feature

# Create PR on GitHub
```

### Fixing a Bug
```bash
# Create branch
git checkout -b fix/bug-description

# Fix the bug
# ... edit files ...

# Add test for the bug
# ... edit tests/test_*.py ...

# Verify fix
make test

# Commit and push
git add .
git commit -m "Fix: bug description"
git push origin fix/bug-description
```

## Testing Locally

### Run specific test
```bash
pytest tests/test_models_regression.py::test_nam_regressor_fit
```

### Run with verbose output
```bash
pytest -v tests/
```

### Run with coverage report
```bash
pytest --cov=nampy --cov-report=html tests/
open htmlcov/index.html  # View coverage report
```

### Test on specific Python version
```bash
# Using pyenv or conda
pyenv install 3.10.0
pyenv local 3.10.0
make test
```

## Troubleshooting

### Import errors
```bash
# Reinstall in development mode
pip uninstall nampy
pip install -e .
```

### Test failures
```bash
# Clear cache and rerun
pytest --cache-clear tests/
```

### Build issues
```bash
# Clean everything
make clean
rm -rf *.egg-info
pip install -e ".[dev]"
```

### Pre-commit issues
```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean
```

## Resources

- **GitHub**: https://github.com/OpenTabular/NAMpy
- **Issues**: https://github.com/OpenTabular/NAMpy/issues
- **Contributing**: See CONTRIBUTING.md
- **Release Process**: See RELEASE_CHECKLIST.md

## Tips

1. **Always format before committing**: `make format`
2. **Run tests frequently**: `make test`
3. **Check coverage**: `make test-cov`
4. **Use pre-commit hooks**: They catch issues early
5. **Write descriptive commit messages**
6. **Add tests for new features**
7. **Update CHANGELOG.md** for user-facing changes

## Need Help?

- Check CONTRIBUTING.md for detailed guidelines
- Open an issue on GitHub
- Ask in GitHub Discussions

---

Happy coding! 🚀

