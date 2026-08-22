# Upstream references

This directory contains the tracked vendored `mgcv` source and local clones of
external reference implementations used for audits and parity investigations.

Only `mgcv/`, this file, and `manifest.json` are intended to be tracked. Other
repositories are ignored and can be recreated with:

```bash
python3 scripts/fetch_upstreams.py
python3 scripts/verify_upstreams.py
```

The fetch command creates shallow clones and writes the resolved commit IDs to
the ignored `lock.json`. External repositories are reference material only;
their code is not imported into the NAMpy runtime. Check each upstream license
before adapting any implementation.

The former repository-root `mgcv/` directory now has one canonical location at
`upstreams/mgcv/`; its contents were not changed during the move. GAM parity
tests should install this source into a temporary R library and set
`MGCV_LIB_PATH` when exact parity against the vendored source is required.
