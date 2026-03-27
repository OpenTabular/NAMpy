from ..basemodels.spam import SPAM
from ..configs.spam_config import DefaultSPAMConfig
from .sklearn_regressor import SklearnBaseRegressor


class SPAMRegressor(SklearnBaseRegressor):
    """
    Scikit-learn compatible regressor using the Scalable Polynomial Additive
    Model (SPAM).

    Wraps the :class:`~nampy.base_models.spam.SPAM` architecture inside
    NAMpy's ``SklearnBaseRegressor`` so that it exposes the standard
    ``fit`` / ``predict`` / ``score`` interface and can be used directly
    with scikit-learn utilities (``GridSearchCV``, ``Pipeline``, etc.).

    SPAM learns a degree-k polynomial via low-rank tensor decompositions
    (Dubey et al., NeurIPS 2022).  Two variants are available through the
    ``use_neural`` flag:

    * **SPAM-LINEAR** (``use_neural=False``, default) — uses geometric
      rescaling :math:`\\tilde{x}_l = \\text{sign}(x) \\cdot |x|^{1/l}`.
      Extremely fast; scales to 100k+ features.
    * **SPAM-NEURAL** (``use_neural=True``) — replaces the rescaled inputs
      with per-feature MLP sub-networks (one per polynomial degree), matching
      the NAM sub-network used in the paper.  More expressive but slower.

    The model returns a rich prediction dict from ``forward()`` that
    includes per-feature unary contributions and pairwise contributions for
    every feature pair when ``degree >= 2``.  These can be accessed after
    fitting via ``model.model.get_feature_importances(...)``.

    Parameters
    ----------
    degree : int, optional
        Maximum polynomial degree k.  ``degree=2`` (default) adds pairwise
        interactions; ``degree=3`` adds triplet interactions.  Interpretability
        degrades beyond pairwise (Section 5 of the paper).
    rank : int or list[int], optional
        Rank(s) of the low-rank tensor decomposition.  A single int is
        broadcast across all degrees ≥ 2; a list of length ``degree - 1``
        sets a rank per degree.  Larger rank → higher capacity but more
        parameters.  Performance typically plateaus at moderate rank
        (see Figure 1B of the paper).  Default is 100.
    use_neural : bool, optional
        If ``False`` (default): SPAM-LINEAR.  If ``True``: SPAM-NEURAL with
        per-feature MLP sub-networks of width ``layer_sizes``.
    layer_sizes : list[int], optional
        Hidden layer widths for each per-feature MLP (SPAM-NEURAL only).
        Default is ``[64, 64]``.
    shared_bases : bool, optional
        For multi-output regression (``num_classes > 1``), share the basis
        vectors U across outputs and learn output-specific singular values λ.
        Reduces parameters from O(2drC) to O((d+r)C + rd).  Default ``True``.
    l1_lambda : float, optional
        L1 sparsity coefficient on the basis vectors.  Drives interaction
        weights toward zero; the paper shows ~6% of pairwise interactions
        suffice for competitive accuracy (Figure 1C).  Default ``0.0``.
    dropout : float, optional
        Dropout applied to singular values λ at each forward pass (basis
        dropout, Section 3.2).  Default ``0.0``.
    feature_dropout : float, optional
        Dropout applied across additive term contributions.  Default ``0.0``.
    intercept : bool, optional
        Whether to learn a global bias term.  Default ``True``.
    numerical_preprocessing : str, optional
        Preprocessing applied to numerical features before they are fed to
        the model (handled by NAMpy's preprocessing pipeline).  Common
        choices: ``"ple"`` (periodic linear encoding), ``"standardization"``,
        ``"minmax"``, ``"quantile"``.  Default ``"ple"``.
    **kwargs
        All remaining keyword arguments are forwarded to
        ``SklearnBaseRegressor`` and then to ``BaseModel``.  This includes
        training-time overrides such as ``lr``, ``weight_decay``,
        ``lr_patience``, ``lr_factor``, and any Lightning ``Trainer``
        arguments accepted by NAMpy.

    Examples
    --------
    >>> from nampy.models import SPAMRegressor
    >>> model = SPAMRegressor(degree=2, rank=100, use_neural=False)
    >>> model.fit(X_train, y_train, max_epochs=100, lr=1e-3)
    >>> preds = model.predict(X_test)

    SPAM-NEURAL with pairwise interactions and L1 sparsity:

    >>> model = SPAMRegressor(
    ...     degree=2,
    ...     rank=200,
    ...     use_neural=True,
    ...     layer_sizes=[64, 64],
    ...     l1_lambda=0.01,
    ... )
    >>> model.fit(X_train, y_train, max_epochs=150)
    """

    def __init__(self, **kwargs):
        super().__init__(model=SPAM, config=DefaultSPAMConfig, **kwargs)