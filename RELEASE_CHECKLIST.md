# NAMpy release checklist

## Prepare

- [ ] Choose the release version and update `nampy/__version__.py`.
- [ ] Add the dated release entry to `CHANGELOG.md`.
- [ ] Confirm README, API docs, support matrices, and examples match the public
      surface.
- [ ] Confirm `GAM_IMPLEMENTED.md` and `GAM_NOT_IMPLEMENTED.md` still describe the
      tested GAM boundary.
- [ ] Confirm the release commit contains no generated API stubs, cached parity
      artifacts, credentials, or local build output.

The package metadata reads its version from `nampy.__version__`; there is no
second version field to edit.

## Validate

Run these configured gates on a clean checkout using Python 3.11 or 3.12:

```bash
ruff check nampy tests
mypy nampy
python scripts/check_repository_hygiene.py
sphinx-build -E -W --keep-going -b html docs docs/_build/html
```

- [ ] Run the smallest relevant neural and GAM release-contract slices locally.
- [ ] Let CI run the complete matrix on Python 3.11 and 3.12.
- [ ] Confirm live `mgcv` parity tests used the vendored `mgcv` package rather
      than untracked snapshot caches.
- [ ] Confirm Linux, macOS, and Windows portability jobs pass.
- [ ] Confirm the documentation job, including notebooks, passes with warnings
      treated as errors.

## Build and inspect

Build in a fresh checkout or empty artifact directory:

```bash
python -m build
twine check dist/*
```

- [ ] Inspect the sdist and wheel file lists.
- [ ] Confirm the wheel includes `nampy/py.typed` and all required subpackages.
- [ ] Install the wheel with its `all` extra into a fresh Python 3.11
      environment.
- [ ] From outside the checkout, run
      `python /path/to/repository/scripts/check_installed_package.py` to verify
      the installed public imports, package metadata, `py.typed`, and GAM exports.

## Publish

- [ ] Merge the reviewed release commit and wait for all required checks.
- [ ] Create and push the signed `vX.Y.Z` tag.
- [ ] Create a GitHub release from that tag.
- [ ] Approve the protected `pypi` environment if required.
- [ ] Verify the trusted-publishing workflow uploads the already-validated
      artifact; do not rebuild it in the publish job.
- [ ] Install the published wheel and verify `nampy.__version__`.
- [ ] If a critical issue is found, yank the affected PyPI release rather than
      deleting it.
