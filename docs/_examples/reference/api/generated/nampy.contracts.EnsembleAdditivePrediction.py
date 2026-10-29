"""Runnable reference example: api/generated/nampy.contracts.EnsembleAdditivePrediction."""

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
result = EnsembleAdditivePrediction(mean=prediction, link_std=np.zeros((n,1)), term_std={'x': np.zeros((n,1))}, intercept_std=0., n_estimators=2)
print(type(result).__name__)
