import pytest

from nampy.preprocessing import Preprocessor

NUMERICAL_MODES = [
    "ple",
    "binning",
    "one_hot",
    "standardization",
    "normalization",
    "quantile",
    "polynomial",
    "splines",
]


@pytest.mark.parametrize("numerical_preprocessing", NUMERICAL_MODES)
def test_preprocessor_numerical_modes(mixed_data, numerical_preprocessing):
    X, y = mixed_data
    preprocessor = Preprocessor(
        numerical_preprocessing=numerical_preprocessing,
        categorical_preprocessing="int",
        n_bins=8,
        degree=2,
        knots=4,
        quantile_preprocessing="feature",
        quantile_output_distribution="normal",
        quantile_n_quantiles=10,
        cat_cutoff=0.1,
    )
    preprocessor.fit(X, y)
    transformed = preprocessor.transform(X)

    assert transformed
    assert all(arr.shape[0] == len(X) for arr in transformed.values())

    cat_info, num_info = preprocessor.get_feature_info()
    assert isinstance(cat_info, dict)
    assert isinstance(num_info, dict)


def test_preprocessor_categorical_one_hot(mixed_data):
    X, y = mixed_data
    preprocessor = Preprocessor(
        numerical_preprocessing="standardization",
        categorical_preprocessing="one_hot",
        cat_cutoff=0.1,
    )
    preprocessor.fit(X, y)
    transformed = preprocessor.transform(X)

    assert any(key.startswith("cat_") for key in transformed.keys())


def test_preprocessor_treat_all_integers_as_numerical(mixed_data):
    X, _ = mixed_data
    preprocessor = Preprocessor(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        cat_cutoff=0.1,
        treat_all_integers_as_numerical=True,
    )
    numerical_features, categorical_features = preprocessor._detect_column_types(X)

    assert "int_cat" in numerical_features
    assert "int_cat" not in categorical_features


def test_preprocessor_decision_tree_bins(mixed_data):
    X, _ = mixed_data
    y = (X["num1"] > 0).astype(int).to_numpy()

    preprocessor = Preprocessor(
        numerical_preprocessing="binning",
        categorical_preprocessing="int",
        use_decision_tree_bins=True,
        n_bins=4,
        task="classification",
        cat_cutoff=0.1,
    )
    preprocessor.fit(X, y)
    transformed = preprocessor.transform(X)

    assert transformed


def test_preprocessor_quantile_global(mixed_data):
    X, y = mixed_data
    preprocessor = Preprocessor(
        numerical_preprocessing="quantile",
        quantile_preprocessing="global",
        quantile_output_distribution="normal",
        quantile_n_quantiles=10,
        cat_cutoff=0.1,
    )
    preprocessor.fit(X, y)
    assert preprocessor.quantile_preprocessor is not None

    transformed = preprocessor.transform(X)
    assert transformed


def test_preprocessor_get_set_params(mixed_data):
    X, y = mixed_data
    preprocessor = Preprocessor(
        numerical_preprocessing="standardization",
        categorical_preprocessing="int",
        n_bins=5,
        knots=3,
        cat_cutoff=0.1,
    )
    params = preprocessor.get_params()
    assert params["n_bins"] == 5
    assert params["knots"] == 3

    preprocessor.set_params(n_bins=7, knots=4, cat_cutoff=0.2)
    assert preprocessor.n_bins == 7
    assert preprocessor.n_knots == 4
    assert preprocessor.cat_cutoff == 0.2


def test_preprocessor_invalid_modes():
    with pytest.raises(ValueError):
        Preprocessor(numerical_preprocessing="bad")
    with pytest.raises(ValueError):
        Preprocessor(categorical_preprocessing="bad")
    with pytest.raises(ValueError):
        Preprocessor(quantile_preprocessing="bad")
    with pytest.raises(ValueError):
        Preprocessor(quantile_output_distribution="bad")
