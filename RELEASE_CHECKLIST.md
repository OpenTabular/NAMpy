# Release Checklist for NAMpy

This checklist should be followed before each release to ensure quality and completeness.

## Pre-Release Checklist

### Code Quality
- [ ] All tests pass locally: `make test`
- [ ] Code is formatted: `make format`
- [ ] Linting passes: `make lint`
- [ ] Type checking passes (or issues documented): `make type-check`
- [ ] Test coverage is adequate: `make test-cov`

### Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md is updated with new version
- [ ] All docstrings are complete and accurate
- [ ] Examples in README work correctly
- [ ] CONTRIBUTING.md reflects current process

### Version Management
- [ ] Version number updated in `nampy/__version__.py`
- [ ] Version number updated in `pyproject.toml`
- [ ] CHANGELOG.md has entry for new version
- [ ] Git tag created: `git tag v0.x.x`

### Testing
- [ ] Fresh virtualenv install test:
  ```bash
  python -m venv test_env
  source test_env/bin/activate
  pip install dist/nampy-*.whl
  python -c "import nampy; print(nampy.__version__)"
  ```
- [ ] Run example code from README
- [ ] Test on multiple Python versions (3.8, 3.9, 3.10, 3.11, 3.12)

### Package Build
- [ ] Clean build artifacts: `make clean`
- [ ] Build package: `make build`
- [ ] Check package: `twine check dist/*`
- [ ] Verify package contents:
  ```bash
  tar -tzf dist/nampy-*.tar.gz | head -20
  ```

### CI/CD
- [ ] All GitHub Actions workflows pass
- [ ] No failing tests in CI
- [ ] Build artifacts generated successfully

## Release Process

### 1. Prepare Release
```bash
# Update version
vim nampy/__version__.py
vim pyproject.toml

# Update changelog
vim CHANGELOG.md

# Commit changes
git add .
git commit -m "Prepare release v0.x.x"
```

### 2. Test Locally
```bash
# Run all quality checks
make quality

# Build package
make build

# Test installation
python -m venv test_env
source test_env/bin/activate
pip install dist/nampy-*.whl
python -c "import nampy; print(nampy.__version__)"
deactivate
rm -rf test_env
```

### 3. Push to GitHub
```bash
# Push changes
git push origin main

# Wait for CI to pass
# Check: https://github.com/OpenTabular/NAMpy/actions

# Create and push tag
git tag v0.x.x
git push origin v0.x.x
```

### 4. Test PyPI (Optional but Recommended)
```bash
# Upload to Test PyPI
make publish-test

# Test installation from Test PyPI
pip install --index-url https://test.pypi.org/simple/ nampy
```

### 5. Publish to PyPI
```bash
# Upload to PyPI
make publish

# Or use GitHub Release (triggers automatic publish)
# Go to: https://github.com/OpenTabular/NAMpy/releases/new
# Create new release with tag v0.x.x
```

### 6. Verify Release
```bash
# Wait a few minutes, then test
pip install nampy
python -c "import nampy; print(nampy.__version__)"
```

### 7. Post-Release
- [ ] Create GitHub Release with changelog
- [ ] Announce on relevant channels
- [ ] Update documentation site (if applicable)
- [ ] Monitor issue tracker for problems

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Incompatible API changes
- **MINOR** (0.1.0): New functionality, backwards compatible
- **PATCH** (0.0.1): Bug fixes, backwards compatible

### Pre-1.0.0 Releases
- Version 0.x.x indicates development/beta status
- Breaking changes allowed in minor versions
- Version 1.0.0 indicates stable public API

## Common Issues

### Build Fails
```bash
# Clean everything and rebuild
make clean
rm -rf *.egg-info
pip install -e ".[dev]"
make build
```

### Import Errors After Install
- Check `__init__.py` has proper imports
- Verify `__all__` exports are correct
- Check for circular imports

### PyPI Upload Fails
- Verify credentials: `~/.pypirc`
- Check package name availability
- Ensure version number is unique

## Emergency Rollback

If a release has critical issues:

```bash
# Yank the release on PyPI (doesn't delete, but prevents new installs)
# Go to: https://pypi.org/manage/project/nampy/releases/

# Or use twine (requires PyPI permissions)
# Note: Yanking is preferred over deletion
```

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPI Help](https://pypi.org/help/)

