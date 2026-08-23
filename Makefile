.PHONY: help install install-dev test test-cov lint type-check hygiene quality docs docs-serve docs-clean clean build publish publish-test

help:
	@echo "NAMpy Development Commands"
	@echo "=========================="
	@echo "install          Install package"
	@echo "install-dev      Install package with dev dependencies"
	@echo "test             Run a targeted test slice: make test TEST=..."
	@echo "test-cov         Run a targeted slice with coverage: make test-cov TEST=..."
	@echo "lint             Run linters (ruff)"
	@echo "type-check       Run type checking (mypy)"
	@echo "hygiene          Check release-sensitive repository state"
	@echo "quality          Run static quality checks"
	@echo "docs             Build documentation"
	@echo "docs-serve       Build and serve documentation locally"
	@echo "docs-clean       Clean documentation build"
	@echo "clean            Clean build artifacts"
	@echo "build            Build distribution packages"
	@echo "publish          Publish to PyPI (use with caution)"

install:
	pip install -e ".[all]"

install-dev:
	pip install -e ".[all,dev]"
	pre-commit install

test:
	@test -n "$(TEST)" || (echo "Specify the smallest relevant slice, for example: make test TEST=tests/neural/test_neural_sklearn_contracts.py"; exit 2)
	pytest $(TEST)

test-cov:
	@test -n "$(TEST)" || (echo "Specify the smallest relevant slice with TEST=..."; exit 2)
	pytest --cov=nampy --cov-report=term-missing --cov-report=html $(TEST)

lint:
	ruff check nampy tests

type-check:
	mypy nampy

hygiene:
	python tests/repository_hygiene.py

quality: lint type-check hygiene
	@echo "Static quality checks passed!"

docs:
	sphinx-build -E -W --keep-going -b html docs docs/_build/html
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
