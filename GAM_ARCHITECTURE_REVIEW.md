# GAM Architecture Review

## Summary

This review covers `nampy/gam/` production code and its tests, with focus on:

- redundancy
- separation of concerns
- subsystem boundary leakage
- stale compatibility debt

`mgcv` parity remains hard constraint. Recommendations below target packaging, ownership, and API cleanup, not algebraic rewrites.

## Findings

### 1. High: facade stack has no clear owner for fit orchestration and backend APIs

Evidence:

- `nampy/gam/__init__.py` re-exports `fit_model_core`, `solve_fit`, and `FitCoreSolution` from `engine` while also exposing `engine` itself as top-level surface.
- `nampy/gam/engine/__init__.py` is a near-total re-export of `nampy/gam/fit/__init__.py`.
- `nampy/gam/fit/__init__.py` re-exports a large mix of orchestrator functions, low-level linear algebra helpers, postprocess hooks, state types, and solver entry points.
- `nampy/gam/model/gam_solve.py` depends on `engine` for orchestration and backend dispatch, but also reaches directly into `smoothing_selection`, `fit.postprocess`, and `compiler`.

Impact:

- `engine` is not a real boundary; it is a second export barrel over `fit`.
- `fit` is both subsystem and compatibility layer, so ownership of solver APIs is unclear.
- `_GAMSolveMixin` becomes service locator for many internals instead of thin model wrapper.
- Cleanup becomes risky because callers can reasonably import from `gam`, `engine`, or `fit` and expect same symbols.

Recommended cleanup:

- Choose one canonical owner for numerical fit APIs. Best candidate: `fit`.
- Reduce `engine` to narrow compatibility shim with explicit deprecation intent, or remove it from top-level `nampy.gam` exports.
- Move `_GAMSolveMixin` toward one boundary call per concern:
  - compile design
  - resolve smoothing config
  - solve fixed-sp fit
  - optimize smoothing
  - assemble result
- Stop adding new symbols to `engine` unless they are intentionally public.

### 2. High: prediction and offset behavior split across model mixins, predict package, and special-case model methods

Evidence:

- `nampy/gam/model/gam_data.py` owns `_coerce_optional_offset`, `_combine_offsets`, `_coerce_offset`, and `_prediction_offset`.
- `nampy/gam/fit/offsets.py` separately owns `coerce_offset_array` and `resolve_prediction_offset`.
- `nampy/gam/predict/predictions.py` implements general single-predictor prediction flow.
- `nampy/gam/predict/linear_predictor_matrix.py` separately rebuilds prediction matrices.
- `nampy/gam/model/api.py` contains separate `_general_family_*` prediction paths for multi-predictor families, including duplicated offset and matrix assembly logic.

Impact:

- Offset semantics have more than one owner.
- Prediction matrix construction exists both in `predict/*` and model methods.
- General-family prediction logic bypasses `predict` package, so fit/predict pairing leaks into model wrapper.
- Future parity fixes in prediction need multiple edits, raising drift risk.

Recommended cleanup:

- Make `predict/` canonical owner of prediction assembly for both single- and multi-predictor families.
- Keep offset coercion/resolution in one module only; model mixin should delegate, not duplicate.
- Introduce one predictor-matrix builder interface that handles both scalar and multi-predictor families.
- Keep model wrapper responsible only for user input coercion and public method shape.

### 3. High: fit-state contracts are duplicated across `fit/state.py` and `engine/state.py`

Evidence:

- `nampy/gam/engine/state.py` defines `FitState` and `PenalizedSystem`.
- `nampy/gam/fit/state.py` imports those types, wraps them in `FitCoreSolution`, and adds additional compatibility behavior.
- `fit/state.py` docstring says "`FitCoreSolution` remains ... during migration", which signals unresolved transition state.
- `fit/__init__.py` re-exports `FitState` and `PenalizedSystem` from `fit/state.py`, but actual definitions live under `engine/state.py`.

Impact:

- State types do not have obvious home.
- `engine.state` sounds canonical, but `fit.state` is actual consumer-facing module.
- Migration wrapper behavior means callers can use attribute forwarding instead of explicit contracts, which hides state ownership and increases accidental coupling.

Recommended cleanup:

- Move canonical dataclass definitions under one owner, preferably `fit/state.py`.
- Leave `engine/state.py` as compatibility import-only shim if needed.
- Tighten `FitCoreSolution` API over time: prefer explicit `.fit_result`, `.fit_state`, `.penalized_system` access over wide `__getattr__` forwarding.

### 4. Medium-High: compiler data contracts leak into unrelated subsystems and become de facto global types

Evidence:

- `nampy/gam/compiler/structures.py` defines both compiler contracts (`CompiledModel`, `CompiledPredictor`, `CompiledTerm`) and generic `PenaltySpec`.
- `nampy/gam/constraints/identifiability.py` imports compiler types directly and mutates predictor-wide compiled structures.
- `nampy/gam/smooths/smooth_base.py` and `nampy/gam/penalties/subsystem.py` import `PenaltySpec` from compiler structures.
- `CompiledPredictor.build_new_matrix()` directly calls `term.smooth.predict_matrix`, so prediction behavior lives inside compiler contract objects.

Impact:

- Compiler package owns structures used by constraints, prediction, penalties, and smooth runtime.
- `PenaltySpec` is not compiler-specific, but its home says it is.
- Compiler objects are not just compiled snapshots; they also execute prediction behavior.
- This blurs runtime term ownership versus compiler ownership.

Recommended cleanup:

- Split generic penalty/runtime contracts from compiler-only structures.
- Keep `Compiled*` objects as assembled design artifacts.
- Move `PenaltySpec` to penalties or smooth runtime layer.
- Decide whether `build_new_matrix()` belongs on compiled objects or on prediction service layer; today it makes compiler a hidden prediction subsystem.

### 5. Medium-High: tests treat broad internal surface as stable API, blocking cleanup

Evidence:

- `tests/test_gam_unit_coverage.py` imports dozens of internal modules directly across `compiler`, `constraints`, `fit`, `families`, `smoothing_selection`, `smooths`, and `specs`.
- Same file asserts import identity for internal symbols, effectively freezing module placement.
- `tests/test_mgcv_additional_scenarios.py` still imports `nampy.gam.design.compiler` and `nampy.gam.formula.preprocess`, which do not match current package layout described in `nampy/gam/__init__.py`.
- Other tests import low-level solver and criterion internals directly instead of through thin test helpers.

Impact:

- Internal refactors will cause widespread test churn even when behavior is unchanged.
- Tests encode packaging decisions, not only behavior/parity.
- Stale path imports suggest compatibility layers or dead references remain ungoverned.

Recommended cleanup:

- Separate behavioral tests from structure/assertion tests.
- Replace broad direct imports with focused test helpers for parity-critical internals.
- Define small intentionally-stable internal test surface if needed, instead of exposing whole tree.
- Remove or explicitly shim stale path references so package layout truth is singular.

### 6. Medium: repository docs and module docstrings still describe old pipeline/module names

Evidence:

- `AGENTS.md` still points to `design/constructors.py` and `design/compiler.py`; repo now uses `compiler/`.
- `nampy/gam/constraints/identifiability.py` docstring references `gam/runtime/factory.py`, `gam/smooths/construct.py`, and `gam/runtime/compile.py`, which do not match current tree.
- `tests/test_mgcv_additional_scenarios.py` still references `gam.design.compiler`.

Impact:

- Contributors cannot trust docs as source of architecture truth.
- Review and cleanup work gets harder because old boundaries remain in prose after code moved.
- Stale names encourage more compatibility shims and accidental duplicate modules.

Recommended cleanup:

- Update one canonical architecture description first: package docstring or repo guide.
- Make module docstrings match actual package names.
- Remove legacy path language unless compatibility layer truly exists and is documented as such.

### 7. Medium: model mixins own too much subsystem logic instead of wrapper-only responsibilities

Evidence:

- `_GAMSolveMixin` performs smoothing parameter normalization, design compilation, penalty assembly, backend capability checks, criterion dispatch, optimizer dispatch, and fit-result assembly.
- `_GAMDataMixin` owns generic feature coercion plus formula-prediction preprocessing plus offset helpers.
- `GAM` object stores very large set of raw internals from every subsystem as direct attributes.

Impact:

- `model/` is not only public wrapper layer; it is active subsystem logic host.
- Internal methods become hidden service API required by fit, predict, and smoothing-selection code.
- Harder to reuse core without full `GAM` instance shape.

Recommended cleanup:

- Keep model layer as wrapper/coordinator over narrower subsystem services.
- Move result assembly and smoothing-resolution policy into dedicated internal services/modules.
- Reduce implicit duck-typed requirements on `model` objects and document required interfaces where unavoidable.

## Cleanup Sequence

1. Freeze intended boundaries in docs.
   - Declare canonical homes for public APIs, fit APIs, prediction APIs, and state contracts.

2. Contain test leakage.
   - Add helper imports or narrow internal test facade.
   - Stop tests from asserting module placement except where intentionally stable.

3. Collapse facade stack.
   - Make `fit` canonical numerical owner.
   - Downgrade `engine` to compatibility shim.

4. Consolidate prediction/offset ownership.
   - One offset module.
   - One prediction assembly path.
   - General-family prediction through `predict/`.

5. Consolidate state contracts.
   - One module defines `FitState` and `PenalizedSystem`.
   - Keep `FitCoreSolution` explicit.

6. Narrow compiler ownership.
   - Move generic contracts out of compiler.
   - Decide whether prediction matrix building stays on compiled artifacts or moves to prediction layer.

## Target Architecture

- `nampy.gam`
  - public user-facing surface only
- `nampy.gam.model`
  - wrapper, user input normalization, public methods
- `nampy.gam.compiler`
  - spec/runtime to compiled design assembly only
- `nampy.gam.constraints`
  - side conditions over compiled artifacts only
- `nampy.gam.fit`
  - canonical numerical solve/state/covariance/postprocess owner
- `nampy.gam.predict`
  - canonical prediction owner for single- and multi-predictor families
- `nampy.gam.smoothing_selection`
  - objective and optimizer internals
- `nampy.gam.engine`
  - temporary compatibility shim only, or removed

## Notes

- No production behavior was changed in this review.
- No tests were run; this was architecture inspection only.
- Recommendations intentionally avoid parity-sensitive algebra changes.
