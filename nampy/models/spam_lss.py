from ..basemodels.spam import SPAM
from ..configs.spam_config import DefaultSPAMConfig
from .sklearn_lss import SklearnBaseLSS


class SPAMLSS(SklearnBaseLSS):
    """
    Distributional regression using the Scalable Polynomial Additive Model
    (SPAM) via NAMpy's Location-Scale-Shape (LSS) framework.

    Wraps :class:`~nampy.base_models.spam.SPAM` inside
    ``SklearnBaseLSS``, giving access to NAMpy's full distributional
    regression pipeline.  Rather than predicting a single conditional mean,
    SPAMLSS models the *full conditional distribution* of the response by
    learning all distribution parameters (location, scale, shape) jointly as
    polynomial functions of the input features.

    All distribution families supported by NAMpy's LSS framework are
    available: ``"normal"``, ``"poisson"``, ``"gamma"``, ``"beta"``,
    ``"studentt"``, ``"negativebinom"``, ``"inversegamma"``,
    ``"dirichlet"``, ``"categorical"``.  The chosen family determines the
    number of distribution parameters, which becomes ``num_classes``
    internally.

    Because SPAM is inherently additive and interpretable, SPAMLSS gives you
    not only uncertainty estimates but also an interpretable decomposition of
    *which features drive each distribution parameter* — unary and pairwise
    contributions are available per distribution parameter from
    ``model.model.get_feature_importances(...)``.

    Parameters
    ----------
    degree : int, optional
        Maximum polynomial degree k.  Default ``2`` (pairwise interactions).
    rank : int or list[int], optional
        Rank(s) of the tensor decomposition.  A single int is broadcast; a
        list of length ``degree - 1`` sets a rank per degree.  Default 100.
    use_neural : bool, optional
        ``False`` (default): SPAM-LINEAR with geometric rescaling.
        ``True``: SPAM-NEURAL with per-feature MLP sub-networks.
    layer_sizes : list[int], optional
        MLP hidden widths for SPAM-NEURAL sub-networks.  Default ``[64, 64]``.
    shared_bases : bool, optional
        Share basis vectors U across distribution parameters.  Strongly
        recommended (default ``True``) when the distribution has multiple
        parameters (e.g., normal: location + scale = 2 outputs).
    l1_lambda : float, optional
        L1 sparsity penalty on basis vectors.  Default ``0.0``.
    dropout : float, optional
        Basis dropout applied to singular values λ.  Default ``0.0``.
    feature_dropout : float, optional
        Feature-level dropout across additive terms.  Default ``0.0``.
    intercept : bool, optional
        Whether to learn a bias/intercept term.  Default ``True``.
    numerical_preprocessing : str, optional
        Preprocessing for numerical features.  Default ``"ple"``.
    **kwargs
        Forwarded to ``SklearnBaseLSS`` and ``BaseModel``.  Includes the
        ``family`` argument (passed to ``fit()``) and training overrides
        (``lr``, ``weight_decay``, etc.).

    Examples
    --------
    Normal distributional regression (predict mean *and* variance):

    >>> from nampy.models import SPAMLSS
    >>> model = SPAMLSS(degree=2, rank=100)
    >>> model.fit(X_train, y_train, max_epochs=150, family="normal")

    Count data with a Poisson family:

    >>> model = SPAMLSS(degree=2, rank=50, l1_lambda=0.01)
    >>> model.fit(X_train, y_train, max_epochs=100, family="poisson")

    SPAM-NEURAL with Student-t for heavy-tailed regression:

    >>> model = SPAMLSS(
    ...     degree=2,
    ...     rank=100,
    ...     use_neural=True,
    ...     layer_sizes=[64, 64],
    ...     shared_bases=True,
    ... )
    >>> model.fit(X_train, y_train, max_epochs=200, family="studentt")

    References
    ----------
    Dubey, A., Radenovic, F., Mahajan, D. (2022).
    *Scalable Interpretability via Polynomials*.
    NeurIPS 2022. https://github.com/facebookresearch/nbm-spam
    """

    def __init__(self, **kwargs):
        super().__init__(model=SPAM, config=DefaultSPAMConfig, **kwargs)