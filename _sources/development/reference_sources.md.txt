# Local upstream references

The repository's ignored `upstreams/` directory contains local clones of
external reference implementations used for audits, implementation work, and
fixture generation.

Nothing below `upstreams/` or in its local maintenance tooling is tracked or
included in a source distribution. External repositories are development-only
reference material. They are not imported by NAMpy and are not required by its
normal test suite or CI. Check each upstream license before adapting any
implementation.

Normal parity tests read versioned static fixtures from
`tests/reference_fixtures/`. Developers refreshing those fixtures must use the
explicit `NAMPY_REFRESH_REFERENCE_FIXTURES=1` mode and record the exact source
version in `tests/reference_fixtures/manifest.json`. Install the selected local
R sources into a temporary library and use `MGCV_LIB_PATH` or `SCAM_LIB_PATH`
to select it during fixture generation.

## Fixture policy

Every fixture is deterministic gzip-compressed JSON. Its content key combines
the reference operation with all inputs that affect the result. A missing
fixture is a test failure in normal mode, even if R or an upstream checkout is
available; this prevents accidental source execution and makes CI behavior
independent of the host machine.

Fixtures are separated by source namespace:

- `mgcv/` contains fits, constructors, prediction, diagnostics, and optimizer
  traces produced by `mgcv` 1.9-4;
- `scam/` contains constrained-basis, fitting, coefficient-transform, and
  diagnostic results produced by SCAM 1.2-22;
- `nbm_spam/` contains dense/sparse NBM, SPAM, and hybrid model tensors; and
- `sian/` contains upstream block-network states and outputs.

The manifest records source versions and commits. Update it whenever a fixture
is regenerated from a different source revision. Fixture changes should be
reviewed as behavioral baseline changes, not as disposable caches.

The initial mgcv corpus preserves the exact historical repository-local 1.9-4
reference tree and records its whole-tree digest. That tree has one known line
difference from the CRAN 1.9-4 mirror in `R/gam.fit3.r` (the sign of `Sstep`).
This is declared in the manifest rather than being misrepresented as a pristine
upstream commit. Any future decision to rebaseline on pristine CRAN mgcv must
regenerate and review the affected fixtures explicitly.

An intentional rebaseline must obtain the exact source separately. Do not mix
fixtures from different source trees under one unchanged provenance record.

## Refresh workflow

Install the local R references when the affected slice needs them, then select
their libraries:

```bash
export MGCV_LIB_PATH="$PWD/.cache/mgcv-lib"
export SCAM_LIB_PATH="$PWD/.cache/scam-lib"
```

Then explicitly enable generation and run only the parity case being added or
updated:

```bash
NAMPY_REFRESH_REFERENCE_FIXTURES=1 pytest path/to/test.py::test_name -v
```

Neural fixture refreshes likewise require the relevant local checkout. They do
not need an R installation. After generation, unset the environment variable
and rerun the same slice; that second run proves the fixture is self-contained.
Review the fixture manifest, fixture diff, and source version together.

`NAMPY_REFRESH_REFERENCE_FIXTURES=1` fills only missing keys and never changes
an existing baseline. For an intentional source-version rebaseline, use
`NAMPY_REBUILD_REFERENCE_FIXTURES=1`; it bypasses and overwrites existing
fixtures exercised by the selected test slice.

Legacy raw JSON caches are unsupported. New cases must write through the shared
fixture helpers rather than create another cache format.
