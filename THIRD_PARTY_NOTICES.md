# Third-party notices

NAMpy contains adaptations of small numerical components from the following
projects. The surrounding NAMpy code remains MIT-licensed; these notices
identify the external source and its license.

## entmax

- Source: https://github.com/deep-spin/entmax
- Used by: `nampy/neural/modules/nodegam_ops.py`
- Upstream license: MIT
- Upstream authors and citation information are preserved by the upstream
  project. The adapted sparsemax/entmax routines should not be represented as
  original NAMpy implementations.

## NODE

- Source: https://github.com/Qwicen/node
- Used by: `nampy/neural/modules/nodegam_utils.py`
- Upstream license: MIT (see the upstream repository's `LICENSE.md`)
- The NODE-derived tree-building routines are adapted for NAMpy's additive
  model interface. The upstream project and its paper should remain credited
  in future redistribution changes.
