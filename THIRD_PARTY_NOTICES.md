# Third-party notices

NAMpy contains adaptations of small numerical components from the following
projects. The surrounding NAMpy code remains MIT-licensed; these notices
identify the external source and its license.

The repositories in `upstreams/` are development-time reference clones and
are not included in the Python distribution. Their individual license files
remain in those upstream checkouts; a clone is not, by itself, an adaptation
or a redistribution of its source.

## entmax

- Source: https://github.com/deep-spin/entmax
- Used by: `nampy/neural/architectures/components/sparse_activations.py`
- Upstream license: MIT
- Upstream authors and citation information are preserved by the upstream
  project. The adapted sparsemax/entmax routines should not be represented as
  original NAMpy implementations.

## NODE

- Source: https://github.com/Qwicen/node
- Used by: `nampy/neural/architectures/components/oblivious_trees.py`, `nampy/neural/architectures/components/additive_trees.py`, and `nampy/neural/architectures/components/term_extraction.py`
- Upstream license: MIT (see the upstream repository's `LICENSE.md`)
- The NODE-derived tree-building routines are adapted for NAMpy's additive
  model interface. The upstream project and its paper should remain credited
  in future redistribution changes.

## IGANN

- Source: https://github.com/MathiasKraus/igann
- Used by: `nampy/neural/architectures/igann.py`
- Upstream license: MIT, copyright MathiasKraus (2022)
- NAMpy adapts the feature-wise ELM construction, ridge boosting sequence,
  linear initialization, and optional ABESS feature-selection design to its
  registry, preprocessing, and additive-output contracts.

## SIAN

- Source: https://github.com/EnouenJ/sparse-interaction-additive-networks
- Used by: ``nampy/neural/interaction_selection/``,
  ``nampy/neural/architectures/components/block_masked_additive.py``, and
  ``nampy/neural/architectures/sian.py``
- Upstream license: the vendored upstream commit does not contain a license
  file; review licensing before redistributing source-derived changes.
- NAMpy independently integrates the published Archipelago/FIS pipeline,
  fractional-heredity search, and block-masked/independent term representation
  behind generic contracts. The upstream project and NeurIPS 2022 paper remain
  the behavioral and attribution references.

## NBM and SPAM

- Source: https://github.com/facebookresearch/nbm-spam
- Used as the behavioral reference for:
  ``nampy/neural/architectures/nbm.py``,
  ``nampy/neural/architectures/spam.py``,
  ``nampy/neural/architectures/nbm_spam.py``, and
  ``nampy/neural/architectures/components/concept_bases.py``.
- Upstream license: Creative Commons Attribution-NonCommercial 4.0.
- NAMpy implements the published shared-basis, sparse active-tuple, low-rank
  polynomial, and combined NBM-SPAM equations behind its own architecture and
  training contracts. The upstream checkout is a development/parity reference
  and is excluded from distributions. Review the upstream non-commercial terms
  and provenance before redistributing source-derived adaptations.

## SCAM

- Source: https://github.com/cran/scam (CRAN mirror), version 1.2-22,
  copyright Natalya Pya (2012-2024).
- Used by: ``nampy/gam/coefficients/transforms.py``,
  ``nampy/gam/splines/shape/``, ``nampy/gam/smooths/shape/``,
  ``nampy/gam/fit/solvers/shape_constrained.py``,
  ``nampy/gam/fit/selection/criteria/shape.py``,
  ``nampy/gam/fit/selection/optimize/shape_bfgs.py``, and the SCAM-specific
  derivative/residual/AR(1) paths.
- Upstream license: GPL version 2 or later, as declared by SCAM's
  ``DESCRIPTION``.  These files include direct behavioral ports of SCAM's
  constructors, constrained Newton solver, GCV/UBRE derivatives, BFGS
  optimizer, prediction, derivative, and AR(1) transformations.  Downstream
  redistribution must preserve the upstream copyright and comply with the
  applicable GPL terms; the repository's general MIT statement must not be
  read as relicensing these source-derived portions.
- Behavioral reference: Pya, N. and Wood, S.N. (2015), “Shape constrained
  additive models,” *Statistics and Computing* 25(3), 543–559.

## Reference-only implementations

The following projects are retained for implementation comparison and feature
audits; no source is copied into the package solely by cloning them:

- Google Research NAM, `google-research/google-research`.
- ProtoNAM, GAMformer/TiCL, NAE, CoxSE/CoxNAM, CRISP-NAM,
  DNAMite, GP-NAM, HNAM, NAM-FS, LA-NAM, GamiNet, HONAM, and neuralGAM.
- pyGAM, InterpretML, CRAN `mgcv`, qgam, Effector, and regional-RHALE.

Consult `upstreams/manifest.json` and `UPSTREAM_LEDGER.md` for exact URLs,
roles, and local paths. Before redistribution, review each upstream license
and preserve its attribution requirements for any future adaptation.
