# Contributing to GenTG-Limb

Thanks for your interest in contributing!

## Getting started
1. **Fork** the repo and create a feature branch: `git checkout -b feat/your-feature`.
2. Install dev dependencies: `pip install -r requirements.txt`.
3. Run linters and tests locally:
   ```bash
   ruff check .
   black --check .
   isort --check-only .
   mypy .
   pytest
   ```

## Development guidelines
- Keep functions **typed** where reasonable.
- Prefer **pure** utils for math; keep I/O (logging, plotting) outside core modules.
- Add or update **unit tests** for all new code.
- Update **docs/** when public APIs change.
- Avoid breaking configs; add a new config if you need different defaults.

## Pull Requests
- Link to the issue and describe what changed.
- Include before/after metrics for training/eval changes (where applicable).
- Green CI is required (format, type-check, tests).

## Code of Conduct
This project follows our [Code of Conduct](CODE_OF_CONDUCT.md).
