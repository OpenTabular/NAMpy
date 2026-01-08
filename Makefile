.PHONY: help install install-dev test test-cov lint format clean build publish docs

help:
	@echo "NAMpy Development Commands"
	@echo "=========================="
	@echo "install          Install package"
	@echo "install-dev      Install package with dev dependencies"
	@echo "test             Run tests"
	@echo "test-cov         Run tests with coverage"
	@echo "lint             Run linters (ruff)"
	@echo "format           Format code (black, isort)"
	@echo "type-check       Run type checking (mypy)"
	@echo "quality          Run all quality checks"
	@echo "docs             Build documentation"
	@echo "docs-serve       Build and serve documentation locally"
	@echo "docs-clean       Clean documentation build"
	@echo "clean            Clean build artifacts"
	@echo "build            Build distribution packages"
	@echo "publish          Publish to PyPI (use with caution)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pre-commit
	pre-commit install

test:
	pytest tests/

test-cov:
	pytest --cov=nampy --cov-report=term-missing --cov-report=html tests/

lint:
	ruff check nampy/ tests/

format:
	black nampy/ tests/
	isort nampy/ tests/

type-check:
	mypy nampy/

quality: format lint type-check test
	@echo "All quality checks passed!"

docs:
	cd docs && make html
	@echo "Documentation built successfully!"
	@echo "Open docs/_build/html/index.html in your browser"

docs-serve: docs
	@echo "Serving documentation on http://localhost:8000"
	@cd docs/_build/html && python -m http.server 8000

docs-clean:
	cd docs && make clean

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*

