## Start here

Read these in order when relevant:
1. `CLAUDE.md` for repository-specific Claude guidance.
2. The vendored upstream `mgcv` R/C sources for parity-sensitive GAM work.
3. `pyproject.toml` for actual tool configuration.

## Core principle

For `nampy/gam/`, this repository is **not** trying to build a loosely inspired GAM library. It is trying to **mirror `mgcv` behavior in Python exactly and as faithfully as possible**.

For parity-sensitive work, the upstream vendored `mgcv` source code in this repository is the primary behavioral specification. Prefer reproducing upstream logic over introducing cleaner-looking or more idiomatic Python formulations.

## Non-negotiable rules

- Do not replace upstream logic with a fresh derivation when the upstream implementation exists in the repo.
- Do not add heuristic, approximate, or best-effort parity fallbacks in parity-sensitive code.
- If a strict `mgcv` port is not yet possible, leave the surface unsupported or add a small explicit TODO rather than shipping heuristic behavior.
- Do not make numerically meaningful algebraic rewrites unless parity tests demonstrate no regression.
- Do not change ordering of penalties, constraints, pivots, side conditions, or block assembly casually.
- Do not use matrix inverses where upstream logic uses solves/factorizations.
- Do not silently broaden unsupported behavior.
- Do not run the full test suite by default.

## Test policy: smallest sufficient slice only

Always validate with the **smallest targeted test slice** that meaningfully covers the change.

Preferred order:
1. exact test function,
2. exact test file,
3. narrow `-k` expression within one file,
4. slightly broader neighboring tests only if the change truly spans them.

Examples:

```bash
pytest tests/test_mgcv_snapshot_parity.py::test_name -v
pytest tests/test_mgcv_snapshot_parity.py -v
pytest tests/test_mgcv_trace_parity.py -k optimizer -v
pytest tests/test_mgcv_pc_id_parity.py -k linked_id -v
```

Avoid unless explicitly justified:

```bash
pytest
pytest tests
python -m pytest
```

If you believe a broader run is necessary, explain why.

## Expected workflow for GAM changes

1. Identify the failing or relevant targeted test first.
2. Identify the exact Python subsystem involved.
3. Locate the corresponding upstream `mgcv` R and/or C routine in the vendored sources.
4. Mirror upstream control flow and data transformations as directly as practical.
5. Run the smallest relevant validation slice.
6. Only expand test coverage if the change crosses subsystem boundaries.

## How to implement parity-sensitive fixes

When porting from upstream `mgcv`:
- preserve control flow structure when possible,
- preserve operand ordering when numerically relevant,
- preserve indexing conventions carefully,
- preserve penalty/block ordering exactly,
- preserve centering / identifiability semantics,
- preserve edge-case branching behavior,
- document the upstream function name in code comments if that mapping is not obvious.

Do not “improve” an algorithm just because a different Python implementation seems more elegant.

## Representation vs behavior parity

- Behavioral parity remains strict: fit, predict, scores, EDF, smoothing-parameter behavior, penalty structure, constraints, and block assembly should match `mgcv`.
- Raw constructor parity should be strict only up to mathematically indeterminate eigenspace orientation. For eigendecomposition-based smooths, prefer canonicalized or invariant comparisons (for example row-space, projector, or penalty-spectrum comparisons) over exact column-by-column basis matching.
- Do not add implementation-level `Rscript` probing, custom LAPACK library selection, or other platform-specific solver hooks solely to force raw basis orientation parity.

## Repository structure

### Neural Additive Models

- `nampy/basemodels/<model>.py` — PyTorch `nn.Module` + Lightning harness
- `nampy/models/<model>.py` — sklearn-style wrappers
- `nampy/configs/<model>_config.py` — config dataclasses
- `nampy/arch_utils/` — shared neural architecture utilities

### GAM subsystem

- `nampy/gam/formula/`, `nampy/gam/specs/` — formula/spec parsing
- `nampy/gam/smooths/`, `nampy/gam/runtime/` — runtime terms, bases, penalties
- `nampy/gam/design/constructors.py` — constructed terms
- `nampy/gam/design/compiler.py` — predictor compilation
- `nampy/gam/constraints/identifiability.py` — side conditions
- `nampy/gam/fit/orchestrator.py` — fitting orchestration
- `nampy/gam/predict/`, `nampy/gam/parity/`, `nampy/gam/diagnostics/` — prediction and diagnostics
- `nampy/splines/` — low-level spline primitives

### Public GAM API

Treat `nampy/gam/__init__.py` exports as the intended public surface:
- `fit_model_core`
- `solve_fit`
- `FitCoreSolution`

## Architectural invariants

- Runtime terms own basis semantics.
- Design code should remain basis-agnostic.
- One canonical owner per concept.
- Fit and predict transforms must remain paired.
- Unsupported inputs should raise explicit errors, not degrade silently.

## Tooling from `pyproject.toml`

Use the repository’s configured tools rather than guessing:

- formatter: `black`
- import sorter: `isort`
- linter: `ruff`
- type checker: `mypy`
- test runner: `pytest`

Typical focused commands:

```bash
black path/to/touched_file.py tests/test_mgcv_snapshot_parity.py
isort path/to/touched_file.py tests/test_mgcv_snapshot_parity.py
ruff check path/to/touched_file.py tests/test_mgcv_snapshot_parity.py
mypy nampy/gam
```

Prefer touched-file or subsystem-scoped checks over sweeping repo-wide passes.

## Testing map

Primary parity-oriented tests live in `tests/`:
- `test_mgcv_snapshot_parity.py`
- `test_mgcv_output_parity.py`
- `test_mgcv_trace_parity.py`
- `test_mgcv_pc_id_parity.py`
- `test_mgcv_known_gaps.py`
- helpers: `mgcv_parity_utils.py`, `mgcv_parity_structure_utils.py`

## What a good final report should contain

When you finish a code change, report:
- what changed,
- which upstream `mgcv` file(s) and function(s) were used as the reference,
- which exact targeted test command(s) were run,
- whether parity improved, held, or remains unresolved,
- any uncertainty or unverified surface area.

## What to avoid in edits

- broad refactors mixed with parity fixes,
- opportunistic renames across many files,
- changing tolerances without cause,
- changing snapshots/baselines without explaining why,
- introducing duplicate implementations of the same concept,
- repo-wide formatting churn.
