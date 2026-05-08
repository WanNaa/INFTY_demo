# Contributing to INFTY

Thank you for your interest in contributing to INFTY. INFTY is a research-oriented optimization toolkit for Continual AI, so contributions should preserve scientific correctness, reproducibility, and compatibility with PyTorch-style training loops.

## Contribution scope

Useful contributions include:

- new continual-learning optimizers;
- improvements to optimizer correctness, speed, or memory usage;
- visualization or analysis utilities;
- reproducible examples;
- tests for optimizer behavior and import compatibility;
- documentation, tutorials, and troubleshooting notes.

## Development setup

```bash
git clone https://github.com/WanNaa/INFTY_demo.git
cd INFTY_demo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[examples]
```

For documentation work:

```bash
python -m pip install -r requirements-docs.txt
mkdocs serve
```

## Coding conventions

- Keep public APIs compatible with PyTorch optimizer usage where possible.
- Optimizer wrappers should accept `params`, `base_optimizer`, `model`, and an `args` dictionary unless there is a strong reason not to.
- Closures should follow the INFTY contract: return `(logits, loss_list)`, where `loss_list` is a list of scalar tensors.
- Avoid hard-coding dataset, benchmark, or task-specific assumptions inside library code.
- Add docstrings to public classes and functions.
- Keep expensive visualization and analysis outputs under a configurable `output_dir`.

## Adding a new optimizer

1. Choose the optimizer family:
   - geometry reshaping: `src/infty/optim/geometry_reshaping/`
   - zeroth-order updates: `src/infty/optim/zeroth_order_updates/`
   - gradient filtering: `src/infty/optim/gradient_filtering/`
2. Implement the optimizer class.
3. Export the class in `src/infty/optim/__init__.py` and update `__all__`.
4. Add a minimal example or test.
5. Update `docs/api_reference.md` and `docs/user_guide.md`.

## Testing

Run the existing tests before opening a pull request:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```

For documentation validation:

```bash
mkdocs build --strict
sphinx-build -b html docs/sphinx docs/sphinx/_build/html
```

## CI/CD overview

GitHub Actions automates the main validation and delivery steps for this repository:

- `CI` runs on every pull request, on pushes to `main`, and on manual dispatch. It executes the test suite across supported Python versions, validates the MkDocs and Sphinx documentation builds, and checks source/wheel packaging with `python -m build` plus `twine check`.
- `Documentation` deploys the MkDocs site to GitHub Pages after successful pushes to `main` when documentation or library files change.
- `Release` runs on version tags that match `v*`. It verifies that the Git tag matches the built package version, publishes distributions to PyPI through trusted publishing, and creates a GitHub release with the built artifacts attached.

The release workflow expects PyPI trusted publishing to be configured for this repository and the `pypi` GitHub environment to exist before the first tagged release.

## Pull request checklist

- [ ] The change has a clear research or engineering motivation.
- [ ] Public APIs are documented.
- [ ] Examples remain runnable.
- [ ] Tests pass locally.
- [ ] Documentation builds without broken links.
- [ ] New optimizer behavior is explained in the user guide or API reference.

## Reporting issues

Please include:

- INFTY version or commit hash;
- Python version;
- PyTorch version;
- CUDA version, if relevant;
- optimizer name and `args` dictionary;
- base optimizer settings;
- closure code or a minimal reproduction script;
- full error traceback.
