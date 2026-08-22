# Debug Scripts

Use this directory for small, purpose-built debugging helpers when a parity
investigation cannot be expressed cleanly as a targeted pytest case.

Rules:

- Prefer a targeted test first.
- Keep each script focused on one bug, one subsystem, or one invariant.
- Note the upstream `mgcv` routine or test surface the script is checking.
- Prefer invariant-based comparisons when raw representation is not uniquely
  determined.
- Remove or trim scripts once the parity issue is understood or covered by a
  lasting test.
