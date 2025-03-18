# CLAUDE.md - Coding Standards & Commands for choice-learn

## Build & Test Commands
- Install: `make install` (conda) or `USE_CONDA=false make install` (venv)
- Install dev: `pip install -r requirements-developer.txt`
- Lint: `ruff check choice_learn/`
- Format: `ruff format choice_learn/`
- Test all: `pytest -n auto tests/`
- Test single: `pytest tests/path/to/test_file.py::test_function`
- Run with coverage: `pytest --cov=choice_learn tests/`
- Serve docs: `make serve_docs_locally`

## Code Style
- Python 3.9+ code base
- Line length: 100 characters
- Docstrings: NumPy style convention
- Quotes: Double quotes
- Imports: Standard library first, then third-party, then choice_learn
- Use type annotations where possible
- Naming: snake_case for functions/variables, CamelCase for classes
- Pre-commit hooks check formatting, types, and security
- Uses ruff for linting/formatting: E, W, F, I, N, D, etc.
