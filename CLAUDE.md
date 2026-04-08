# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Before running any command, do 

```conda activate nampy```

to activate the appropriate conda environment.

Run a single test file:
```bash
pytest tests/test_mgcv_snapshot_parity.py -v
pytest tests/test_mgcv_snapshot_parity.py::test_name -v
```

## Architecture

NAMpy is an interpretable tabular ML framework with two distinct subsystems:

### 1. Neural Additive Models (`nampy/basemodels/`, `nampy/models/`)

Each model (NAM, GPNAM, NBM, NATT, NAMformer, NodeGAM, SplineNAM, QNAM, SNAM, TreeNAM, LinReg) follows a layered pattern:
- **`nampy/basemodels/<model>.py`** — PyTorch `nn.Module` + Lightning harness (`TaskModel`)
- **`nampy/models/<model>.py`** — scikit-learn-compatible wrappers (`<Model>Regressor`, `<Model>Classifier`, `<Model>LSS`)
- **`nampy/configs/<model>_config.py`** — hyperparameter dataclasses
- **`nampy/arch_utils/`** — shared building blocks (MLP layers, normalization, attention, embeddings)

Three task flavors per model: regression, classification, distributional regression (LSS). All expose `.fit(X, y)`, `.predict(X)`, `.score(X, y)`.

### 2. GAM Subsystem (`nampy/gam/`)

A Python reimplementation of R's `mgcv`. **`nampy/gam/ARCHITECTURE.md` is the canonical source of truth** for design decisions. Our goal is that our results should match with the results of mgcv to machine precision. The fit pipeline has 7 stages:

| Stage | Location | Role |
|-------|----------|------|
| 1. Formula/spec | `gam/formula/`, `gam/specs/` | Parse TermSpec objects |
| 2. Runtime terms | `gam/smooths/`, `gam/runtime/` | Fit basis & penalties; own basis semantics |
| 3. Term wrapper | `gam/design/constructors.py` | ConstructedTerm (constraints, by-variable) |
| 4. Predictor compilation | `gam/design/compiler.py` | Assemble CompiledPredictor |
| 5. Side conditions | `gam/constraints/identifiability.py` | Column deletion, centering |
| 6. Model fitting | `gam/fit/orchestrator.py` | Solve coefficients, optimize smoothing |
| 7. Prediction/diagnostics | `gam/predict/`, `gam/parity/`, `gam/diagnostics/` | Inference, parity checks |

**Public API** (`nampy/gam/__init__.py` exports only): `fit_model_core`, `solve_fit`, `FitCoreSolution`

**Low-level basis primitives** live in `nampy/splines/` and are consumed by runtime terms in `gam/smooths/`.

**Key design rules** (enforced by architecture):
- Runtime terms own all basis semantics — design code must be basis-agnostic
- One canonical owner per concept (no duplicated logic across files)
- Fit and predict transforms must be paired (no hidden one-off transforms)
- Explicit errors for unsupported inputs; no silent approximations

### Key Data Flow (GAM)

`TermSpec` → `RuntimeTerm` (fitted basis, penalties) → `ConstructedTerm` → `CompiledPredictor` → `fit_model_core()` → `FitCoreSolution` (coefficients, smoothing params, covariance, EDF)

## Testing

`tests/` default collection is now mgcv parity-only:
- `test_mgcv_snapshot_parity.py` — broad numeric parity vs. R's mgcv
- `test_mgcv_output_parity.py` — predictions and model-comparison outputs
- `test_mgcv_trace_parity.py` — smoothing optimizer trace parity
- `test_mgcv_pc_id_parity.py` — `pc=` and linked-`id=` parity
- `test_mgcv_known_gaps.py` — tracked strict parity mismatches
- `mgcv_parity_utils.py`, `mgcv_parity_structure_utils.py` — shared test helpers

Legacy characterization / guardrail files remain in `tests/legacy_mgcv_*.py` but are not part of default collection.

Parity snapshots compare against R mgcv output; do not break these without understanding the numerical implications.

## Code Quality

- **Line length**: 88 (black)
- **Linter**: ruff (rules E, W, F, I, C, B; E501 ignored)
- **Type checker**: mypy (non-strict; major deps have `ignore_missing_imports`)
- Pre-commit hooks enforce black + isort + ruff before commit
