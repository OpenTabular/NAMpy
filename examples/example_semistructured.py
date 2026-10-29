"""Joint fixed-penalty spline and identifiable neural representation."""

import numpy as np
import pandas as pd

from nampy import (
    SemiStructuredLSS,
    SemiStructuredRegressor,
    SemiStructuredTrainingConfig,
)
from nampy.neural.configs import DefaultSemiStructuredConfig

rng = np.random.default_rng(67)
X = pd.DataFrame({"x": rng.uniform(-1, 1, 100), "z": rng.uniform(-1, 1, 100)})
y = 0.5 + np.sin(3 * X.x) + 0.6 * X.x * X.z + rng.normal(0, 0.1, 100)
architecture = DefaultSemiStructuredConfig(
    structured_formula="~s(x,k=5)+z",
    neural_features=("x", "z"),
    hidden_dims=(8,),
    representation_dim=4,
)
training = SemiStructuredTrainingConfig(
    epochs=40,
    lr=0.03,
    patience=15,
    smoothing_params=[0.2],
    random_state=18,
)
for estimator in (SemiStructuredRegressor, SemiStructuredLSS):
    model = estimator(architecture_config=architecture, training=training)
    model.fit(X.iloc[:80], y.iloc[:80], X_val=X.iloc[80:], y_val=y.iloc[80:])
    components = model.predict_components(X)
    components.validate_additive_reconstruction()
    print(estimator.__name__, model.orthogonality_error_, model.predict(X.iloc[:3]))
    # The fitted training projection is reused even for a single query row.
    np.testing.assert_allclose(
        model.predict(X.iloc[:1]), model.predict(X)[:1], atol=1e-12
    )
