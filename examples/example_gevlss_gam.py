"""Generalized extreme value GAM for location, log scale and shape."""

import numpy as np
import pandas as pd

from nampy.gam import GAM
from nampy.gam.families.gamlss.gevlss import gevlss


def fit_example():
    rng = np.random.default_rng(261027)
    x = np.linspace(-1, 1, 180)
    location = 2.5 + 1.4*np.sin(3*x)
    scale = np.exp(-.5+.15*x)
    shape = .1
    y = location + ((-np.log(rng.uniform(size=len(x))))**(-shape)-1)*scale/shape
    data = pd.DataFrame({'x': x, 'y': y})
    model = GAM(family=gevlss(),
                formula=['y ~ s(x, bs="cr", k=6)', '~ 1', '~ 1'],
                smoothing_method='REML', optimize_smoothing=True)
    model.fit(data=data)
    return data, model


if __name__ == '__main__':
    data, model = fit_example()
    print('Source response columns (location, log scale, shape):')
    print(model.predict(data.iloc[:5]))
    eta = model.predict(data.iloc[:5], type='link')
    print('Natural parameters (location, scale, shape):')
    print(model.family.distribution_parameters_from_eta(eta))
