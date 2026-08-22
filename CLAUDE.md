## Primary objective

For the GAM subsystem, the goal is **behavioral parity with `mgcv`**, not a merely reasonable Python approximation. When working on `nampy/gam/`, treat the vendored upstream **R and C `mgcv` sources in this repository as the primary specification**. Mirror the same logic, ordering, control flow, constraints, and edge-case behavior in Python whenever practical.

Do **not**:

- rederive `mgcv` from papers or memory,
- “clean up” numerics by changing algebra/order of operations without evidence,
- replace an upstream routine with a more idiomatic approach unless parity requires it and tests confirm it.
- add heuristic, approximate, or best-effort parity fallbacks in parity-sensitive code,
- add NAMpy-only optimizer rescue heuristics after an upstream-style endpoint has been found,
- use finite-difference outer-derivative fallbacks for parity-sensitive `ml` / `reml` / `laml` paths,
- rewrite unsupported formula constructs into approximate fallback specs,
- partially support linked `id=` groups by silently pooling only a compatible subset.
- investigate parity-sensitive behavior with one-off shell or REPL snippets when the same check can be encoded as a targeted test,
- rely on ephemeral exploratory commands when a small script under `debug/` would preserve the probe.

If behavior differs between an apparent design preference and upstream `mgcv`, prefer upstream parity.

## Test execution policy

Do **not** run the full test suite by default.

Always run the **smallest targeted test slice** that can validate the change:

1. exact test function,
2. exact test file,
3. narrow `-k` selection within one file,
4. only then a slightly broader local slice if needed.

Preferred examples:

```bash
pytest tests/parity/test_mgcv_snapshot_parity.py::test_name -v
pytest tests/parity/test_mgcv_snapshot_parity.py -v
pytest tests/parity/test_mgcv_output_parity.py -k linked_id -v
```

Avoid broad commands such as:

```bash
pytest
pytest tests
python -m pytest
```

Run broader coverage only when clearly justified by the scope of the change.

## Investigation policy

When you need to inspect a failing parity case:

1. prefer adding or refining a narrow pytest case that reproduces the mismatch,
2. prefer invariant-based assertions when the `mgcv` representation is mathematically non-unique,
3. if a test would be too awkward for the probe, create a small focused script under `debug/` and run that instead,
4. avoid ad hoc experiments that are not captured in the repository.

## Architecture

NAMpy has two numerical backends plus shared surfaces:

- `nampy/_contracts.py` — the small backend-neutral contracts (`FeatureSchema`, `AdditivePrediction`); imports neither backend.
- `nampy/plotting/` — backend-neutral term-plot renderer consuming prepared plot-data dicts.
- Ownership rules: `nampy/gam` imports nothing from `neural/` or `models/`, and contains zero torch. PreTab appears only under `neural/` and `models/`.

### 1. Neural backend (`nampy/neural/`, `nampy/models/`)

Each model (NAM, GPNAM, IGANN, NBM, NATT, NAMformer, NodeGAM, SplineNAM, QNAM, SNAM, TreeNAM, LinReg) follows a layered pattern:

- `**nampy/neural/architectures/**` — PyTorch architectures, one file per model; reusable building blocks in `**nampy/neural/architectures/components/**` (BaseModel, MLPs, normalization, attention, embeddings, interactions, sparse activations, oblivious/additive tree blocks)
- `**nampy/neural/objectives.py**` — architecture-independent output, target,
  loss, and metric semantics; `**nampy/neural/task.py**` — Lightning harness;
  `**nampy/neural/contracts.py**` — forward-output key grammar
- `**nampy/neural/registry.py**` — canonical architecture definitions and capabilities
- `**nampy/neural/data/**` — `NAMpyDataModule`/`NAMpyDataset` (PreTab-to-Torch; preprocessor fit on training rows only; offset channel)
- `**nampy/neural/distributions/**` — Torch LSS families and metrics
- `**nampy/neural/configs/<model>_config.py**` — hyperparameter dataclasses
- `**nampy/models/<model>.py**` — registry-generated estimator families
  (`<Model>Regressor`, `<Model>Classifier`, `<Model>LSS`);
  `nampy/models/gam.py` holds the GAM adapters

Three task flavors per model: regression, classification, distributional regression (LSS). All expose `.fit(X, y)`, `.predict(X)`, `.score(X, y)`. Estimators use hand-written `score()` and `__sklearn_tags__` (no sklearn mixin classes — keep it that way).

### 2. GAM subsystem (`nampy/gam/`)

A Python reimplementation of R's `mgcv`. The objective is that results as well as the code should match `mgcv` to machine precision whenever feasible.

The fit pipeline has 7 stages:


| Stage                     | Location                                          | Role                                         |
| ------------------------- | ------------------------------------------------- | -------------------------------------------- |
| 1. Formula/spec           | `gam/formula/`, `gam/specs/`                      | Parse `TermSpec` objects                     |
| 2. Runtime terms          | `gam/smooths/`, `gam/splines/`                    | Fit basis and penalties; own basis semantics |
| 3. Term construction      | `gam/compiler/construct.py`                       | Build `CompiledTerm` objects and maps        |
| 4. Predictor compilation  | `gam/compiler/compile_predictors.py`, `compile_model.py` | Assemble predictors and the compiled model |
| 5. Side conditions        | `gam/constraints/identifiability.py`              | Column deletion, centering                   |
| 6. Model fitting          | `gam/fit/orchestrator.py`                         | Solve coefficients, optimize smoothing       |
| 7. Prediction/diagnostics | `gam/predict/`, `gam/parity/`, `gam/diagnostics/` | Inference, parity checks                     |


**Public API** (`nampy/gam/__init__.py` exports only): `GAM`, `fit_model_core`, `solve_fit`, `FitCoreSolution`

**Low-level basis primitives** live in `gam/splines/` and are consumed by runtime terms in `gam/smooths/`.

### Key design rules

- Runtime terms own all basis semantics — design code must be basis-agnostic.
- One canonical owner per concept — avoid duplicated logic across files.
- Fit and predict transforms must be paired — no hidden one-off transforms.
- Explicit errors for unsupported inputs — no silent approximations.
- Upstream `mgcv` reference code is authoritative for parity-sensitive behavior.
- If strict parity is not implemented yet, raise a clear error or add a small explicit TODO instead of applying a local fallback.

### Key data flow (GAM)

`TermSpec` → `BaseSmoothTerm` (fitted basis, penalties) → `CompiledTerm` →
`CompiledPredictor` / `CompiledModel` → `fit_model_core()` →
`FitCoreSolution` (coefficients, smoothing parameters, covariance, EDF)

## Working rules for `mgcv` parity changes

When changing `nampy/gam/`:

- locate the corresponding upstream `mgcv` R and/or C implementation in the vendored reference sources,
- mirror the upstream routine as directly as possible in Python,
- preserve operation ordering when numerically relevant,
- keep shape conventions, constraints, penalty ordering, and side-condition handling aligned with upstream,
- add comments only where they clarify the mapping from upstream logic,
- compare parity up to `mgcv`-relevant invariants whenever raw representation is not uniquely determined.

For any parity-sensitive change, your final summary should name:

- the upstream file(s) consulted,
- the upstream function(s) mirrored,
- the exact targeted test command(s) run,
- any remaining known gap or uncertainty.

## Testing

`tests/` default collection is mgcv parity-focused:

- `test_mgcv_snapshot_parity.py` — broad numeric parity vs. R `mgcv`
- `test_mgcv_output_parity.py` — predictions and model-comparison outputs
- `tests/optimization/test_mgcv_score_hist_trace_parity.py` — score-history and trace I/O parity
- `tests/optimization/test_mgcv_outer_optimization_parity.py` — full outer-object and optimizer-row parity
- `tests/optimization/test_mgcv_inner_trace_parity.py` — PIRLS and negbin inner-trace parity
- `test_mgcv_pc_id_parity.py` — `pc=` and linked-`id=` parity
- `test_mgcv_known_gaps.py` — tracked strict parity mismatches
- `mgcv_parity_utils.py`, `mgcv_parity_structure_utils.py` — shared test helpers

Parity snapshots compare against R `mgcv` output; do not update or break these casually.

## Code quality

- **Line length**: 88 (`black`)
- **Linter**: `ruff` (rules `E`, `W`, `F`, `I`, `C`, `B`; `E501` ignored)
- **Type checker**: `mypy` (non-strict; major deps have `ignore_missing_imports`)
- Pre-commit hooks enforce `black` + `isort` + `ruff` before commit

Use focused quality checks for touched files whenever possible rather than sweeping repo-wide rewrites.
