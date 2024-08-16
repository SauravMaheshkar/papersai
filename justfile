default:
  @just --list

# Install dependencies
requirements:
  python -m pip install -U uv
  uv pip install -e. "[dev]"
  pre-commit install

# Delete all compiled files and cache
clean:
  find . -type f -name "*.py[co]" -delete
  find . -type f -name "__pycache__" -delete
  rm -rf .mypy_cache/
  rm -rf .pytest_cache/
  rm -rf .ruff_cache/
  rm -rf artifacts/

# Testing
test:
  pytest -vv .

# Basic linting
lint:
  black papersai tests
  ruff check papersai tests
  mypy papersai
