# Copilot Instructions for LFPykit

## Project Overview

`LFPykit` is a Python library providing freestanding implementations of electrostatic forward models for computing extracellular potentials and related measures (EEG, MEG, LFP, CSD) from multicompartment neuron models. It is designed to be used independently of any specific neural simulation software (NEURON, Arbor, LFPy, etc.).

The core design principle is that each forward model class exposes a `get_transformation_matrix()` method returning a 2D linear transformation matrix **M** such that measurements **Y** = **M** @ **I**, where **I** are transmembrane currents.

## Repository Structure

```
LFPykit/
├── .github/                  # GitHub configuration (workflows, templates, instructions)
│   ├── workflows/            # CI/CD GitHub Actions workflows
│   ├── CONTRIBUTING.md       # Contribution guidelines
│   └── ISSUE_TEMPLATE/       # Issue templates
├── lfpykit/                  # Main package source code
│   ├── __init__.py           # Package exports
│   ├── cellgeometry.py       # CellGeometry base class
│   ├── models.py             # LinearModel and derived forward model classes
│   ├── lfpcalc.py            # Low-level LFP calculation routines
│   ├── eegmegcalc.py         # EEG/MEG forward model classes
│   ├── version.py            # Package version
│   └── tests/                # Unit tests (pytest)
├── doc/                      # Documentation source (Sphinx)
├── examples/                 # Example Jupyter notebooks
├── requirements.txt          # pip dependencies (legacy)
├── environment.yml           # conda environment
├── pyproject.toml            # Poetry package configuration
└── README.md                 # Project documentation
```

## Key Architecture Patterns

- **`CellGeometry`** (`cellgeometry.py`): Represents the geometry of a multicompartment neuron (x, y, z coordinates and diameter `d` of each segment).
- **`LinearModel`** (`models.py`): Abstract base class for all forward models. Subclasses must implement `get_transformation_matrix()`.
- All classes (except `CellGeometry`) must implement `get_transformation_matrix()` returning a NumPy array of shape `(n_points, n_inputs)`.
- EEG/MEG classes in `eegmegcalc.py` operate on current dipole moments (output of `CurrentDipoleMoment.get_transformation_matrix()`).

## Physical Units Convention

| Quantity | Unit |
|---|---|
| Transmembrane currents | nA |
| Spatial coordinates | µm |
| Voltages / potentials | mV |
| Extracellular conductivities | S/m |
| Current dipole moments | nA µm |
| Magnetic fields | nA/µm |

There are no explicit runtime checks for units — callers are responsible for consistency.

## Coding Conventions

- Follow **PEP 8** style guidelines for all Python code.
- Line length limit: **127 characters** (as configured in the flake8 workflow).
- Use descriptive variable names and add docstrings to all public classes and methods.
- Use NumPy-style docstrings.
- Imports: standard library first, then third-party (`numpy`, `scipy`, etc.), then local package imports.
- All new model classes must subclass `LinearModel` (or `Model`) and implement `get_transformation_matrix()`.
- Keep one idea/feature per pull request.

## Installation

The project uses [Poetry](https://python-poetry.org/) for dependency management and packaging (Python >= 3.10 required).

```bash
# Install with Poetry
poetry install

# Install with pip (from source)
pip install .

# Install with optional test/doc extras
pip install ".[tests]"
pip install ".[docs]"
```

## Linting

```bash
# Check for syntax errors and undefined names (must pass — zero tolerance)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Full style check (warnings allowed)
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

## Running Tests

```bash
# Run the full test suite
py.test -v

# Run a specific test file
py.test lfpykit/tests/test_module.py -v

# Run a specific test
py.test lfpykit/tests/test_module.py::TestLinearModel -v
```

Tests are located in `lfpykit/tests/` and use `pytest`. New features must include corresponding unit tests.

## Dependencies

- `python >= 3.10`
- `numpy >= 1.15.2` — array operations and linear algebra
- `scipy >= 1.5.2` — special functions and numerical routines
- `sympy` — symbolic mathematics (used in some analytical derivations, optional)
- `MEAutility >= 1.5.1` — multi-electrode array utilities

Dependencies are managed via `pyproject.toml`.

## Contributing

- Read `.github/CONTRIBUTING.md` before submitting changes.
- Always link PRs to a related open issue.
- Ensure all tests pass (`py.test -v`) and linting is clean before opening a PR.
- Use the PR template in `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`.
