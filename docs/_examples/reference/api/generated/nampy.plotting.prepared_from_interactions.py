"""Runnable reference example: api/generated/nampy.plotting.prepared_from_interactions."""

import numpy as np
import pandas as pd
from nampy.contracts import AdditivePrediction, AdditiveTermMetadata, EnsembleAdditivePrediction, FeatureSchema
from nampy.inspection import center_additive_prediction, explain_additive_prediction, term_importance_table

n = 12
X = pd.DataFrame({'x': np.linspace(0, 1, n)})
term = np.sin(3 * X.x.to_numpy())
link = 0.5 + term
prediction = AdditivePrediction(link=link,
                               terms={'x': term}, intercept=0.5, backend='gam',
                               term_metadata={'x': AdditiveTermMetadata(features=('x',))})
prediction.validate_additive_reconstruction()
from nampy.plotting import prepared_from_interactions
X['z'] = np.cos(np.arange(n))
terms = {'x:z': (X.x * X.z).to_numpy()}
result = prepared_from_interactions(X, terms, num_bins=4)
print(type(result).__name__)
