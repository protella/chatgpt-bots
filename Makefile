.PHONY: help install test test-verbose test-coverage test-unit test-integration test-fast test-report clean lint format check

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install all dependencies including test requirements"
	@echo "  make test          - Run all tests with coverage"
	@echo "  make test-verbose  - Run tests with verbose output"
	@echo "  make test-coverage - Run tests and generate HTML coverage report"
	@echo "  make test-unit     - Run only unit tests"
	@echo "  make test-integration - Run only integration tests"
	@echo "  make test-fast     - Run tests without coverage (faster)"
	@echo "  make test-report   - Open coverage report in browser"
	@echo "  make test-watch    - Run tests in watch mode (auto-rerun on changes)"
	@echo "  make clean         - Remove test artifacts and cache"
	@echo "  make lint          - Run linting checks"
	@echo "  make format        - Auto-format code"
	@echo "  make check         - Run all checks (lint + test)"

# Install dependencies
install:
	python3 -m pip install -U -r requirements.txt

# Basic test run with coverage
test:
	python3 -m pytest --cov=. --cov-report=term-missing --cov-report=html --cov-config=.coveragerc -v

# Verbose test output
test-verbose:
	python3 -m pytest --cov=. --cov-report=term-missing --cov-report=html --cov-config=.coveragerc -vv

# Generate detailed coverage report
test-coverage:
	python3 -m pytest --cov=. --cov-report=term-missing --cov-report=html --cov-report=xml --cov-config=.coveragerc
	@echo "Coverage report generated in htmlcov/index.html"

# Run only unit tests
test-unit:
	python3 -m pytest tests/unit --cov=. --cov-report=term-missing --cov-config=.coveragerc -v

# Run only integration tests
test-integration:
	python3 -m pytest tests/integration --cov=. --cov-report=term-missing --cov-config=.coveragerc -v

# Fast test run without coverage
test-fast:
	python3 -m pytest -v

# Open coverage report in browser
test-report:
	@python3 -c "import webbrowser; webbrowser.open('htmlcov/index.html')" 2>/dev/null || echo "Open htmlcov/index.html in your browser"

# Watch mode for continuous testing
test-watch:
	python3 -m pytest-watch --clear --runner "python3 -m pytest --cov=. --cov-report=term-missing -v"

# Clean test artifacts
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf tests/__pycache__
	rm -rf tests/*/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Linting (requires additional packages)
lint:
	@which ruff >/dev/null 2>&1 && ruff check . || echo "Install ruff for linting: pip install ruff"
	@which mypy >/dev/null 2>&1 && mypy . || echo "Install mypy for type checking: pip install mypy"

# Format code
format:
	@which black >/dev/null 2>&1 && black . || echo "Install black for formatting: pip install black"
	@which isort >/dev/null 2>&1 && isort . || echo "Install isort for import sorting: pip install isort"

# Run all checks
check: lint test