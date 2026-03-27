from ..basemodels.spam import SPAM
from ..configs.spam_config import DefaultSPAMConfig
from .sklearn_classifier import SklearnBaseClassifier


class SPAMClassifier(SklearnBaseClassifier):
    """
    Scikit-learn compatible classifier using the Scalable Polynomial Additive
    Model (SPAM).

    Wraps the :class:`~nampy.base_models.spam.SPAM` architecture inside
    NAMpy's ``SklearnBaseClassifier`` to expose the standard
    ``fit`` / ``predict`` / ``predict_proba`` / ``score`` interface and full
    compatibility with scikit-learn utilities.

    For **binary classification** set ``num_classes=1`` (default); the model
    outputs a single logit and the classifier applies a sigmoid threshold.
    For **multi-class classification** set ``num_classes=C`` where C is the
    number of classes; outputs are softmax-normalised logits.

    When ``degree >= 2`` and ``shared_bases=True`` (the default for
    multi-class), the basis vectors U are shared across classes while the
    singular values λ are learned per class, reducing the parameter count
    from O(2drC) to O((d+r)C + rd) — see Section 3.2 of the paper.

    Parameters
    ----------
    degree : int, optional
        Maximum polynomial degree k.  ``degree=2`` (default) models pairwise
        feature interactions.  The paper shows second-order interactions match
        or beat DNN performance on most datasets (Table 1).
    rank : int or list[int], optional
        Rank(s) of the tensor decomposition.  Broadcasts a single int to all
        degrees ≥ 2, or accepts a list of length ``degree - 1``.  Default 100.
    use_neural : bool, optional
        ``False`` (default): SPAM-LINEAR with geometric rescaling.
        ``True``: SPAM-NEURAL with per-feature MLP sub-networks.
    layer_sizes : list[int], optional
        Hidden layer widths for SPAM-NEURAL sub-networks.  Default ``[64, 64]``.
    shared_bases : bool, optional
        Share basis vectors U across classes (multi-class only).  Default
        ``True``.  Strongly recommended when the number of classes is large.
    l1_lambda : float, optional
        L1 sparsity penalty coefficient on basis vectors.  Induces sparse
        pairwise interactions — ~6% of pairs suffice for competitive accuracy
        on concept-bottleneck tasks (Figure 1C of the paper).  Default ``0.0``.
    dropout : float, optional
        Dropout probability applied to singular values λ (basis dropout).
        Default ``0.0``.
    feature_dropout : float, optional
        Dropout probability applied across additive term contributions.
        Default ``0.0``.
    intercept : bool, optional
        Whether to learn a global bias/intercept.  Default ``True``.
    numerical_preprocessing : str, optional
        Preprocessing strategy for numerical features.  Options include
        ``"ple"``, ``"standardization"``, ``"minmax"``, ``"quantile"``.
        Default ``"ple"``.
    **kwargs
        Forwarded to ``SklearnBaseClassifier`` and ``BaseModel``.  Includes
        training overrides (``lr``, ``weight_decay``, etc.) and any Lightning
        ``Trainer`` arguments accepted by NAMpy.

    Examples
    --------
    Binary classification with SPAM-LINEAR:

    >>> from nampy.models import SPAMClassifier
    >>> model = SPAMClassifier(degree=2, rank=100)
    >>> model.fit(X_train, y_train, max_epochs=100)
    >>> probs = model.predict_proba(X_test)

    Multi-class with SPAM-NEURAL and shared bases:

    >>> model = SPAMClassifier(
    ...     num_classes=10,
    ...     degree=2,
    ...     rank=150,
    ...     use_neural=True,
    ...     shared_bases=True,
    ... )
    >>> model.fit(X_train, y_train, max_epochs=150, lr=1e-3)

    References
    ----------
    Dubey, A., Radenovic, F., Mahajan, D. (2022).
    *Scalable Interpretability via Polynomials*.
    NeurIPS 2022. https://github.com/facebookresearch/nbm-spam
    """

    def __init__(self, **kwargs):
        super().__init__(model=SPAM, config=DefaultSPAMConfig, **kwargs)