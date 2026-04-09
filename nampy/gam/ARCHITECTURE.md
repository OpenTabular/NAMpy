# GAM Architecture

This document defines the architecture for the `nampy.gam` subsystem and the surrounding spline/runtime support code.

It is intended to do four things:

1. define **canonical ownership** of each concern,
2. make the **fit → compile → predict** pipeline explicit,
3. establish **invariants** that all new code must follow,
4. provide a stable target for cleanup, refactoring, and mgcv parity work.

This is the source of truth for how the GAM subsystem should be organized going forward.

---

# 1. Scope and goals

The `nampy.gam` package is a Python reimplementation of core `mgcv` concepts:

- smooth term specification and construction,
- basis setup and design-matrix assembly,
- penalty definition and smoothing-parameter bookkeeping,
- identifiability constraints and side conditions,
- runtime term objects for fit and prediction,
- parity tooling against `mgcv`,
- user-facing prediction, summaries, and diagnostics.

The primary goals are:

- **mgcv semantic parity where implemented**
- **clear ownership boundaries**
- **minimal redundancy**
- **predictable transform flow**
- **testable, basis-agnostic compilation**
- **stable public API over an evolving internal architecture**

Non-goals for this layer:

- general-purpose neural spline components,
- broad distributional-regression families unrelated to GAM core,
- experimental utilities that do not participate in GAM fit/predict/parity.

Those may exist elsewhere in the repository, but they are not part of the GAM architectural core.

---

# 2. Design principles

## 2.1 One owner per concept

Every concept must have exactly one canonical implementation.

Examples:

- penalty normalization: one canonical subsystem
- tensor basis algebra: one canonical module
- predictor-wide side conditions: one canonical module
- parametric linear term runtime: one canonical class

Re-export shims are acceptable for public convenience, but they must not contain competing logic.

## 2.2 Runtime terms own basis semantics

Basis-specific behavior belongs in runtime term classes only.

Runtime terms own:

- feature resolution
- basis construction
- term-local constraints
- by-variable handling
- penalty definitions
- new-data transforms
- term-local metadata

No design or prediction module should reimplement basis-specific mathematics.

## 2.3 Design code is basis-agnostic

Design/compilation code assembles already-materialized runtime terms into predictor-level structures.

It may know:

- basis matrices,
- penalty specs,
- coefficient slices,
- transforms,
- smoothing parameter maps.

It must not know:

- how a thin-plate basis is built,
- how tensor marginal centering works,
- how MRF neighborhoods are turned into penalties,
- how factor smooth internals are parameterized.

## 2.4 Fit-time and predict-time transforms must agree exactly

Any transform applied to coefficients or basis columns during fitting must be represented explicitly and reused during prediction.

There must be no hidden, one-off transform logic at prediction time.

## 2.5 Constraints and penalties move together

Whenever a coefficient transform `T` is applied:

- design must become `B @ T`
- every penalty must become `T.T @ S @ T`

This is a hard invariant.

## 2.6 Prefer explicit unsupported behavior over partial behavior

If a piece of `mgcv` semantics is not yet implemented, raise a clear `NotImplementedError`.

Do not approximate silently when doing so risks breaking parity or architecture.

---

# 3. Canonical module ownership

This section defines which module family owns which concern.

## 3.1 `gam/smooths/*` — runtime term implementations

Canonical owner of all runtime term semantics.

Subareas:

- `gam/smooths/univariate/*`
- `gam/smooths/tensor/*`
- `gam/smooths/categorical/*`

Responsibilities:

- fit-time basis construction
- fit-time penalty construction
- term-local constraint absorption
- term-local by-variable handling
- term-local metadata
- prediction matrix generation for new data
- exposing normalized penalty definitions

These classes are the canonical implementation of GAM term behavior.

## 3.2 `gam/design/*` — predictor compilation and assembly

Canonical owner of design assembly.

Responsibilities:

- turning term specs into constructed terms
- attaching linked-basis metadata
- collecting basis blocks across terms
- building compiled predictor structures
- storing coefficient slices, smoothing-parameter maps, and transforms

Design code must be basis-agnostic.

## 3.3 `gam/constraints/*` — coefficient-space linear algebra

Canonical owner of:

- coefficient transforms
- null-space basis extraction from explicit constraints
- predictor-wide identifiability side conditions
- column independence detection
- generic “apply linear constraint” helpers

This package owns **linear algebra over already-built bases**, not basis construction.

## 3.4 `gam/penalties/*` — penalty normalization and metadata

Canonical owner of:

- penalty symmetrization
- penalty eigendecomposition
- rank/null-space inference
- null-space selection penalties
- smoothing override merging
- automatic smoothing-id generation
- normalized `PenaltySpec` semantics

All penalty metadata rules live here.

## 3.5 `gam/formula/*` — formula-side preprocessing only

Canonical owner of:

- formula data extraction
- factor-by expansion planning/materialization
- parametric expansion planning/materialization
- hidden-column reconstruction rules for prediction

This layer must not contain runtime basis logic.

## 3.6 `gam/predict/*` — prediction API consumers

Canonical owner of:

- lpmatrix building
- link/response/term prediction APIs
- term contribution assembly

Prediction code is a consumer of fitted design/runtime state and must not duplicate fit-time logic.

## 3.7 `gam/parity/*` — parity tooling

Canonical owner of:

- parity snapshots
- snapshot comparison
- optimizer trace serialization

Parity tooling must sit on top of a fitted model/core. It is not part of the fitting path.

## 3.8 `gam/diagnostics/*` — reporting and plotting consumers

Canonical owner of:

- summary text formatting
- plotting helpers
- trace display wrappers

Diagnostics consume model state. They must not mutate fitting state or encode alternative semantics.

## 3.9 `gam/terms/*` — public re-exports only

This package exists for public API ergonomics.

It may re-export canonical runtime term classes, but it must not contain alternate implementations.

If a class exists in both `gam/terms/*` and `gam/smooths/*`, one must be removed.

## 3.10 `splines/*` — low-level basis/setup primitives

Canonical owner of basis-specific numerical primitives that are shared by runtime terms.

Examples:

- thin-plate basis setup
- Gaussian-process setup
- MRF neighborhood/penalty helpers
- P-spline low-level basis helpers

These modules should not know about compiled predictors, term specs, or fit results.

---

# 4. End-to-end architecture

The full GAM pipeline should be understood as the following stages.

## Stage 1 — Formula/spec layer

Inputs:

- parsed formula or manually constructed predictor/term specs
- raw feature data
- optional offsets
- optional preprocessing state

Main modules:

- `gam/formula/extract.py`
- `gam/formula/preprocess.py`
- `gam/specs` (not shown here, but conceptually upstream)

Outputs:

- canonical `LinearPredictorSpec` and `TermSpec` objects
- preprocessed working data
- preprocessing state needed for prediction-time reconstruction

## Stage 2 — Runtime materialization

Main modules:

- `gam/runtime/factory.py`
- `gam/smooths/*`

Responsibilities:

- instantiate each term spec into a runtime term object
- resolve basis type / special (`s`, `te`, `ti`, `t2`, etc.)
- fit term-local basis and penalties

Outputs:

- fitted runtime terms exposing:
  - `basis_train`
  - `transform_new`
  - `get_penalty_definitions`
  - constraint/by metadata

## Stage 3 — Term construction wrapper

Main module:

- `gam/design/constructors.py`

Responsibilities:

- adapt runtime terms into `ConstructedTerm`
- apply wrapper-level by handling only if runtime delegated it
- absorb explicit constraints only if runtime delegated them
- preserve metadata describing where transformations occurred

Outputs:

- `ConstructedTerm` objects containing:
  - final term-local training basis
  - prediction function
  - penalty specs
  - term-local transforms/metadata

## Stage 4 — Predictor compilation

Main module:

- `gam/design/compiler.py`

Responsibilities:

- assemble constructed term basis blocks
- assign coefficient slices
- assign smoothing-parameter indices/ids
- normalize penalty specs
- collect term and predictor metadata

Outputs:

- `CompiledPredictor`

At this stage, basis blocks are term-local-final but not yet predictor-globally cleaned for side conditions.

## Stage 5 — Predictor-wide side conditions

Main module:

- `gam/constraints/identifiability.py`

Responsibilities:

- enforce predictor-wide identifiability relative to the current accumulator
- optionally account for intercept span
- transform term bases and penalties consistently
- record which columns survived
- preserve the mapping from runtime coefficient space to fitted coefficient space

Outputs:

- a final `CompiledPredictor`
- a side-condition report

## Stage 6 — Model fitting

The fitting core (outside the files shown here) consumes:

- compiled design matrices
- penalties
- smoothing-parameter map
- family/link/offset information

Outputs include:

- coefficients
- smoothing parameters
- covariance matrices
- EDF and diagnostics
- fitted result object

## Stage 7 — Prediction and parity

Consumers of fitted state:

- `gam/predict/*`
- `gam/parity/*`
- `gam/diagnostics/*`

They must use the compiled design/runtime transform path exactly as stored.

---

# 5. Core abstractions and their meaning

## 5.1 `TermSpec`

A declarative description of one term before runtime materialization.

It may describe:

- parametric term
- smooth term
- special type (`s`, `te`, `ti`, `t2`)
- typed `smooth_spec` union for basis-specific arguments
- by-variable
- smoothing id
- label
- metadata

`TermSpec` is not fitted and owns no numerical state.

## 5.2 Runtime term

A fitted term object from `gam/smooths/*`.

Canonical runtime interface:

- `fit(X, feature_names)`
- `basis_train`
- `transform_new(X_new)`
- `get_penalty_definitions()`
- `label`
- `basis_name`
- `term_type`
- `feature`
- term-local metadata / constraint/by state

This is the canonical owner of basis semantics.

## 5.3 `ConstructedTerm`

A wrapper-level fitted term representation.

It exists to bridge runtime semantics into design assembly.

It may include:

- runtime object
- final train design matrix after wrapper handling
- penalty specs
- prediction function
- explicit fit/predict constraint metadata
- prediction offset
- constructor metadata

## 5.4 `PenaltySpec`

A normalized description of one penalty component.

It should always mean:

- a symmetric penalty matrix
- smoothing-id association
- semantic kind (`smooth`, `null_space`, `random_effect`, etc.)
- rank/null-space metadata
- optional fixed/estimated smoothing override
- stable metadata for parity/debugging

## 5.5 `CompiledTerm`

A predictor-level placed term.

It should always mean:

- this term occupies `coef_slice` in the compiled design
- `basis_train` is the predictor-level training block after any side-condition processing
- `basis_transform` maps runtime coefficient space into final fitted coefficient space
- `kept_columns` / `deleted_columns` describe the final survival map
- `smoothing_indices` identify associated smoothing parameters

## 5.6 `CompiledPredictor`

A fully assembled predictor-level object.

It should always mean:

- `design_matrix` is the final predictor design used for fitting
- compiled terms and penalties align with that design
- `build_new_matrix(X_new)` produces new-data design blocks using the same coefficient-space conventions
- smoothing overrides and maps are stable and final

---

# 6. Invariants

These invariants must hold everywhere.

## 6.1 Basis/penalty shape alignment

For any term or compiled block:

- `basis_train.shape[1] == penalty.shape[0] == penalty.shape[1]`
- if multiple penalties exist, all penalties have the same coefficient dimension for that term/block

## 6.2 `CompiledTerm.basis_transform` is canonical

`basis_transform` is the full coefficient transform from the runtime term’s native coefficient space to the final fitted coefficient space for that compiled term.

Prediction must use this transform and only this transform.

## 6.3 Penalties track all coefficient transforms

If a term basis is transformed from `B` to `B @ T`, every associated penalty must be transformed from `S` to `T.T @ S @ T`.

Subsetting columns without prior transformation is only valid when the basis transform is exactly a column selector in the current coefficient space.

## 6.4 Mixed-type feature matrices are allowed

The GAM system may operate on object arrays when some columns are categorical/factor-like.

Therefore:

- runtime terms must not coerce the entire feature matrix to `float64` unless the subsystem guarantees all columns are numeric
- a runtime term must only coerce the columns it actually needs

## 6.5 Runtime and wrapper handling must not both apply the same transformation

For any concern (by handling, explicit constraints), exactly one layer may apply it:

- runtime term, or
- construction wrapper

Metadata must record which layer handled it.

## 6.6 Exempt terms still span predictor space

Terms exempt from predictor-wide column deletion (e.g. random effects / certain factor smooths) still contribute to the current span when evaluating later terms.

Exemption means “do not delete columns from this term”, not “pretend this term does not exist”.

## 6.7 Zero-width terms are a deliberate policy decision

The codebase must choose one policy and enforce it consistently:

- either allow zero-width compiled terms for bookkeeping,
- or drop them from the final compiled predictor.

The preferred policy is to drop zero-width terms from the final compiled design once behavior is fully stabilized.

## 6.8 Public re-export modules contain no business logic

Files under `gam/terms/*`, `gam/diagnostics/trace.py`, parity shims, etc. must either:

- re-export canonical symbols only, or
- be removed.

---

# 7. Responsibilities by package

## 7.1 `gam/runtime/factory.py`

Purpose:

- convert declarative term specs into runtime term objects

Rules:

- may dispatch on `special`, `bs`, and basis options
- must not implement basis construction directly
- must not duplicate public/runtime class definitions
- must return canonical runtime classes only

## 7.2 `gam/design/compiler.py`

Purpose:

- compile fitted term blocks into a predictor

Rules:

- basis-agnostic
- must normalize penalty specs through penalty subsystem helpers
- must assign stable smoothing ids and smoothing indices
- must not apply predictor-wide side conditions itself

## 7.3 `gam/design/constructors.py`

Purpose:

- bridge runtime terms into `ConstructedTerm`

Rules:

- wrapper-level by handling only if runtime did not do it
- wrapper-level explicit constraint absorption only if runtime did not do it
- no basis-specific mathematics beyond generic linear transforms
- preserve provenance in `constructor_metadata`

## 7.4 `gam/constraints/transforms.py`

Purpose:

- reusable linear algebra utilities

Rules:

- purely generic
- no GAM-family semantics beyond matrix transforms
- safe to use in any term/predictor linear-algebra path

## 7.5 `gam/constraints/identifiability.py`

Purpose:

- predictor-wide side conditions

Rules:

- this is the canonical implementation of global side conditions
- no reverse imports from design alias modules
- must preserve coefficient-transform consistency
- must update both design blocks and penalties coherently

## 7.6 `gam/penalties/algebra.py`

Purpose:

- pure penalty matrix algebra

Rules:

- symmetrization
- eigendecomposition
- null-space penalty construction
- no higher-level smoothing-id semantics

## 7.7 `gam/penalties/subsystem.py`

Purpose:

- penalty metadata and normalization semantics

Rules:

- canonical home for normalized `PenaltySpec` behavior
- all penalty rank/null-space metadata should flow through here
- default smoothing-id rules belong here

## 7.8 `gam/predict/*`

Purpose:

- user-facing prediction behavior

Rules:

- use compiled design/runtime state only
- must not reconstruct fit-time transforms ad hoc
- must support link/response/terms/lpmatrix consistently

## 7.9 `gam/parity/*`

Purpose:

- serialize/compare model fit and predictions against external references

Rules:

- no influence on fit path
- consume fitted core/model API only
- parity-specific recomputations belong under a parity namespace in snapshots

---

# 8. Supported semantics and extension policy

This section is architectural, not exhaustive documentation.

## 8.1 Currently supported smooth families should remain explicit

Support should remain explicit for implemented families such as:

- 1D cubic regression smooths (`cr`, `cs`, `cc`)
- P-splines (`ps`)
- thin plate / shrinkage thin plate (`tp`, `ts`)
- Gaussian process (`gp`)
- tensor products (`te`, `ti`, `t2`) in the currently implemented marginal subset
- random effects (`re`)
- MRF (`mrf`)
- factor smooths (`fs`, `sz`)

Each implemented family must expose the same runtime interface and fit/predict invariants.

## 8.2 Unsupported combinations must remain deliberate

Examples of architecture-friendly behavior:

- raise `NotImplementedError` for unsupported `select=True` combinations
- raise `NotImplementedError` for unsupported `pc` paths
- raise `NotImplementedError` for unsupported tensor marginal bases
- raise `NotImplementedError` when shared-basis linkage is only implemented for a subset

Do not silently fall back to a different semantics.

## 8.3 Adding a new smooth family

A new family should usually require:

1. a runtime term in `gam/smooths/*`
2. low-level basis helpers in `splines/*` if needed
3. factory dispatch registration
4. parity/characterization tests
5. summary/prediction validation if output semantics differ

It should not require changes to:

- predictor compiler internals,
- prediction API,
- generic constraint transforms,

unless the family introduces genuinely new global semantics.

---

# 9. Metadata policy

Metadata is useful, but it must not become an uncontrolled dump.

## 9.1 Metadata categories

Allowed metadata categories:

- provenance:
  - source spec
  - constructor/runtime handling choices
- parity/debug:
  - basis options
  - resolved feature names
  - null-space dimensions
  - rank
- prediction reconstruction:
  - preprocess recipe details

Avoid storing arbitrary large arrays in metadata unless required for prediction or parity.

## 9.2 Required metadata provenance for transformed terms

When a term is altered by wrapper or constraint logic, metadata should record:

- whether by handling was done in runtime or wrapper
- whether constraint absorption was done in runtime or wrapper
- number of absorbed constraints when applicable
- side-condition-deleted original columns
- whether predictor-wide centering was absorbed

This is essential for debugging parity mismatches.

---

# 10. Prediction architecture

Prediction must follow the fitted design, never reconstruct it differently.

## 10.1 `build_new_matrix` is the predictor-level entry point

`CompiledPredictor.build_new_matrix(X_new)` is the canonical predictor-level design builder for new data.

It must:

- call each compiled term’s prediction path,
- apply `basis_transform`,
- concatenate in compiled order.

It must not:

- re-derive fit-time constraints,
- invent new column deletions,
- bypass stored transforms.

## 10.2 Term-level prediction must mirror fit-time coefficient space

Runtime term `transform_new(X_new)` returns prediction matrices in the runtime term’s native fitted coefficient parameterization.

Any later predictor-level transform is applied through `CompiledTerm.basis_transform`.

This division must remain strict.

## 10.3 Offsets

Offsets should be treated in two distinct categories:

- model/predictor-level offsets
- term-level prediction offsets (only if truly needed)

If term-level prediction offsets are kept in the architecture, they must be fully integrated into prediction assembly. Half-integrated offset abstractions are not allowed.

---

# 11. Parity architecture

Parity is a first-class validation layer, not an afterthought.

## 11.1 Types of parity

We distinguish:

- **structural parity**
  - basis dimensions
  - penalty counts
  - smoothing-id layout
  - design shapes

- **fit parity**
  - coefficients
  - smoothing parameters
  - EDF
  - scale/deviance/RSS

- **prediction parity**
  - response/link predictions
  - term contributions
  - lpmatrix

- **optimization-trace parity**
  - criterion path
  - gradients/hessians
  - accepted steps
  - rank diagnostics

## 11.2 Snapshot contract

Parity snapshots should contain:

- semantic fit state from `fit_result.to_dict()`
- prediction outputs
- parity-only recomputations under a separate `parity` section

Parity recomputations must never pollute the core fit serialization contract.

## 11.3 Comparison policy

Comparators must compare:

- scalar fit quantities
- array-valued quantities
- covariance objects when requested
- parity criterion metadata when present

Parity failures should be maximally diagnosable.

---

# 12. Testing strategy

## 12.1 Characterization tests before refactor

Before architectural cleanup, lock down current supported behavior with characterization tests.

These should cover:

- term construction
- compiled design shapes
- penalty counts/shapes
- smoothing-id maps
- predictions for core term families

## 12.2 Unit tests at ownership boundaries

Tests should be organized by owner.

Examples:

### Runtime term tests
- basis shape
- penalty definitions
- new-data transform consistency

### Constraint tests
- coefficient transform correctness
- penalty transform correctness
- side-condition column deletion logic

### Compiler tests
- smoothing-id assignment
- override merging
- compiled slices and term maps

### Prediction tests
- lpmatrix shape/content
- terms/link/response shape consistency
- covariance-based standard errors

### Parity tests
- snapshot build/load/compare
- optimizer trace serialization
- mgcv fixture comparisons

## 12.3 Explicit unsupported-behavior tests

Unsupported paths should be tested to raise the intended error class/message family.

This prevents accidental drift into inconsistent partial implementations.

---

# 13. Public API policy

## 13.1 Public-facing imports should be stable

Users may import via public convenience modules such as:

- canonical smooth modules under `nampy.gam.smooths.univariate/*`
- canonical smooth modules under `nampy.gam.smooths.tensor/*`
- canonical smooth modules under `nampy.gam.smooths.categorical/*`

Internally, however, these should only re-export canonical runtime implementations.

## 13.2 Internal imports should target canonical owners

Internal code should import from the canonical module, not from public re-export shims.

For example:

- internal tensor helpers should import from `gam/basis/tensor.py`
- internal penalty logic should import from `gam/penalties/*`
- internal side-condition logic should import from `gam/constraints/identifiability.py`

This keeps the dependency graph clean and avoids cycles.

---

# 14. Current cleanup targets implied by this architecture

These are architecture-level cleanup requirements.

## 14.1 Remove duplicate implementations

There must be exactly one canonical implementation of:

- `LinearTerm`
- global side-condition logic
- tensor helper logic
- penalty normalization semantics

## 14.2 Convert alias modules into pure shims or remove them

Files that exist only for API compatibility must either:

- contain pure re-exports, or
- be removed.

They must not contain real logic.

## 14.3 Move all penalty semantics behind the penalty subsystem

Any code that infers penalty rank/null-space metadata ad hoc should be migrated toward the canonical penalty subsystem.

## 14.4 Normalize runtime term interface across all families

Every runtime term should satisfy the same fitted-state contract.

This makes compiler, predictor, and parity code simpler and more stable.

---

# 15. Coding rules for contributors

These rules should be followed in all new GAM-core code.

## 15.1 Do not add basis-specific logic outside runtime terms or low-level spline modules

If the code needs to know a basis family’s mathematics, it belongs in:

- `gam/smooths/*`, or
- `splines/*`

not in design/predict/parity code.

## 15.2 Do not duplicate transforms between fit and predict

If a transform is needed at prediction time, it must be stored explicitly from fitting.

## 15.3 Do not coerce full feature matrices unless guaranteed numeric

Only coerce the columns required for the current operation.

## 15.4 Keep metadata informative but bounded

Metadata should help explain provenance and parity, not become an uncontrolled serialization of internals.

## 15.5 Prefer small reusable helpers for repeated patterns

Examples worth centralizing:

- by-state resolution
- smoothing-id generation
- null-space selection penalty construction
- hidden formula-column naming
- prediction matrix assembly

## 15.6 Preserve error clarity

Unsupported behavior should fail early with explicit messages.

---

# 16. Recommended near-term architecture roadmap

This is the implementation order the architecture expects.

## Step 1
Freeze current supported behavior with characterization tests and parity fixtures.

## Step 2
Remove duplicate implementations and reverse alias dependencies.

## Step 3
Make `constraints/identifiability.py` the single canonical implementation of predictor-wide side conditions.

## Step 4
Ensure `CompiledTerm.basis_transform` is the complete, canonical fit→predict transform.

## Step 5
Standardize runtime term interfaces and mixed-type prediction handling.

## Step 6
Centralize penalty metadata and null-space logic behind the penalty subsystem.

## Step 7
Reduce `gam/terms/*` and similar modules to logic-free public shims.

## Step 8
Keep parity, diagnostics, and prediction as pure consumers of fitted state.

---

# 17. Summary

The core architectural idea is simple:

- **runtime terms own basis semantics**
- **design code assembles**
- **constraints code transforms**
- **penalty code normalizes**
- **prediction/parity/diagnostics consume fitted state**
- **public API modules re-export only**

If future code follows that rule set, the GAM subsystem will remain maintainable while continuing to grow toward deeper `mgcv` parity.
