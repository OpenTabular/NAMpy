"""Multinomial GAM with two log-odds predictors and three response classes."""

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.families.gamlss.multinom import multinom


def fit_example():
    rng = np.random.default_rng(261026)
    x = np.linspace(-1, 1, 180)
    eta = np.column_stack([1.7*np.sin(2*x), 1.7*np.sin(3*x)+.2])
    probabilities = np.column_stack([np.ones(len(x)), np.exp(eta)])
    probabilities /= probabilities.sum(axis=1)[:, None]
    y = np.array([rng.choice(3, p=row) for row in probabilities])
    data = pd.DataFrame({'x': x, 'y': y})
    model = GAM(family=multinom(K=2),
                formula=['y ~ s(x, bs="cr", k=6)', '~ s(x, bs="cr", k=6)'],
                smoothing_method='REML', optimize_smoothing=True)
    model.fit(data=data)
    return data, model


if __name__ == '__main__':
    data, model = fit_example()
    print('Class probabilities, baseline category zero first:')
    print(model.predict(data.iloc[:5]))
