from .sklearn_regressor import SklearnBaseRegressor
from .sklearn_classifier import SklearnBaseClassifier
from .sklearn_lss import SklearnBaseLSS
from ..basemodels.treenam import BoostedNAM
from ..configs.boostednam_config import DefaultBoostedNAMConfig


class TreeNAMRegressor(SklearnBaseRegressor):
    """
    Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the NAM model
    with the default NAM configuration.

    The accepted arguments to the NAMRegressor class include both the attributes in the DefaultNAMConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    layer_sizes : list, default=(128, 128, 32)
        Sizes of the layers in the NAM.
    activation : callable, default=nn.SELU()
        Activation function for the NAM layers.
    skip_layers : bool, default=False
        Whether to skip layers in the NAM.
    dropout : float, default=0.5
        Dropout rate for regularization.
    norm : str, default=None
        Normalization method to be used, if any.
    use_glu : bool, default=False
        Whether to use Gated Linear Units (GLU) in the NAM.
    skip_connections : bool, default=False
        Whether to use skip connections in the NAM.
    batch_norm : bool, default=False
        Whether to use batch normalization in the NAM layers.
    layer_norm : bool, default=False
        Whether to use layer normalization in the NAM layers.
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'binning', 'one_hot', 'standardization', and 'normalization'.
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
    cat_cutoff : float or int, default=0.03
        Indicates the cutoff after which integer values are treated as categorical.
        If float, it's treated as a percentage. If int, it's the maximum number of
        unique values for a column to be considered categorical.
    treat_all_integers_as_numerical : bool, default=False
        If True, all integer columns will be treated as numerical, regardless
        of their unique value count or proportion.
    degree : int, default=3
        The degree of the polynomial features to be used in preprocessing.
    knots : int, default=12
        The number of knots to be used in spline transformations.

    Notes
    -----
    - The accepted arguments to the NAMRegressor class are the same as the attributes in the DefaultNAMConfig dataclass.
    - NAMRegressor uses SklearnBaseRegressor as the parent class. The methods for fitting, predicting, and evaluating the model are inherited from the parent class. Please refer to the parent class for more information.

    See Also
    --------
    mambular.models.SklearnBaseRegressor : The parent class for NAMRegressor.

    Examples
    --------
    >>> from mambular.models import NAMRegressor
    >>> model = NAMRegressor(layer_sizes=[128, 128, 64], activation=nn.ReLU())
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> model.evaluate(X_test, y_test)
    """

    def __init__(self, **kwargs):
        super().__init__(model=BoostedNAM, config=DefaultBoostedNAMConfig, **kwargs)
