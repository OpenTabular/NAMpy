For transformed single-predictor ordinary-family (SCAM) models, fixed
smoothing and automatic GCV/UBRE selection are supported. Outer routes include
BFGS, EFS, ``optim``, ``nlm``, and ``nlm.fd``; coefficient optimization supports
Newton and BFGS. Availability depends on the criterion, family, and term
structure. Automatic SCAM ML/REML/LAML remains unsupported.

Fixed smoothing supports transformed terms in multi-predictor/LSS models;
automatic smoothing for that combination requires transformed Laplace
derivatives and remains guarded. Bivariate ``by`` terms, model-level
``select=True`` for transformed models, and user-supplied constraint matrices
remain restricted.

For ordinary Gaussian identity models with unknown scale, AR(1)
ML/REML/LAML uses the bam determinant correction. This does not extend the
SCAM transformed-coefficient likelihood route or non-Gaussian correlated
likelihoods. See :doc:`Compatibility </compatibility>` for guarded
criterion/optimizer combinations. Regression coverage lives in
``tests/optimization/test_mgcv_remaining_routes.py``.
