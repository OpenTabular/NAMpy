"""Runnable reference example: statistical_reference/estimators/SurvivalGAM."""

import numpy as np
import pandas as pd
from nampy.models import SurvivalGAM
from nampy.survival import (SurvivalTarget, CensoringDistribution, brier_score,
                            concordance_index, survival_curve, survival_residuals,
                            prepare_survival_effects, prepare_survival_curve_interval,
                            prepare_survival_effect_interval, combine_survival_predictions,
                            prepare_survival_hazard_interval, prepare_survival_rmst_interval,
                            bootstrap_survival)

X = pd.DataFrame({'x': np.linspace(-1., 1., 30)})
target = SurvivalTarget(
    time=np.r_[np.linspace(.1, .9, 15), np.linspace(1.1, 2.9, 15)],
    event=np.tile([1, 0, 1], 10), entry=np.full(30, .02),
)
model = SurvivalGAM(cuts=[0, 1, 3], formula='~ x', optimize_smoothing=False)
model.fit(X, target)
prediction = model.predict_distribution(X)
result = {'survival': prediction.survival([0, .5, 1, 2, 3]),
          'hazard': prediction.hazard([.5, 2]), 'rmst': prediction.rmst(3),
          'components': model.predict_components(X), 'log_likelihood': model.score(X, target)}
# Descriptive curves and residuals support entry. Evaluation below uses a
# separate cohort without entry; censoring is supplied from training outcomes.
result['residuals'] = survival_residuals(prediction, target)
result['observed_curve'] = survival_curve(target)
evaluation = SurvivalTarget(np.linspace(.15, 2.95, 30), np.tile([0, 1, 1], 10))
censoring = CensoringDistribution.fit(SurvivalTarget([.1, 1., 4.], [1, 0, 1]))
errors = brier_score(prediction, evaluation, [.5, 1., 2.], censoring=censoring)
result['brier'] = errors.score
result['integrated_brier'] = errors.integrated()
result['concordance'] = concordance_index(
    evaluation, -prediction.rmst(2), higher_risk=True)
result['effects'] = prepare_survival_effects(model, X, times=[0, .5, 1, 2, 3])
print({key: type(value).__name__ for key, value in result.items()})
result['curve_interval'] = prepare_survival_curve_interval(model, X, times=[0, .5, 1, 2, 3], n_draws=64, seed=42)
result['simultaneous_band'] = prepare_survival_curve_interval(model, X, times=[0, .5, 1, 2, 3], n_draws=64, seed=42, interval='simultaneous')
result['effect_intervals'] = prepare_survival_effect_interval(model, X, times=[0, .5, 1, 2, 3], reference_X=X.iloc[::2], n_draws=64, seed=42, interval='simultaneous')
result['hazard_interval'] = prepare_survival_hazard_interval(model, X, times=[.5, 1, 2, 3], n_draws=64, seed=42, interval='simultaneous')
result['rmst_interval'] = prepare_survival_rmst_interval(model, X, horizons=[0, 1, 2, 3], n_draws=64, seed=42, interval='simultaneous')
bootstrap = bootstrap_survival(model, X, target, X.iloc[:4], resample_indices=np.stack([np.arange(30), np.r_[np.arange(29), 28]]), effects={'times': [0, .5, 1, 2, 3], 'reference_X': X.iloc[::2]})
result['subject_bootstrap'] = bootstrap.percentile_interval(grid=[0, .5, 1, 2, 3])

# Two prescribed refits demonstrate the API; use a substantial resampling budget
# for inference. Coverage here is joint only over the declared finite grid.
result['bootstrap_band'] = bootstrap.simultaneous_interval(grid=[0, .5, 1, 2, 3])
result['bootstrap_effects'] = bootstrap.effect_interval(interval='simultaneous')
# The same observation contract supports censored parametric time laws.
from tempfile import TemporaryDirectory
from pathlib import Path
for likelihood in ('weibull', 'lognormal'):
    parametric = SurvivalGAM(likelihood=likelihood, formula='~ x', optimize_smoothing=False)
    parametric.fit(X, target)
    latent = parametric.predict_distribution(X)
    with TemporaryDirectory() as directory:
        saved = parametric.save_model(Path(directory) / 'survival.nampy')
        restored = type(parametric).load_model(saved)
    result[likelihood] = {'survival': latent.survival([0, 1, 3]),
                          'parameters': dict(latent.distribution.parameters),
                          'rmst': latent.rmst(3),
                          'remaining_rmst': latent.rmst(3, conditional_on=target.entry),
                          'residuals': survival_residuals(latent, target),
                          'log_likelihood': parametric.score(X, target)}
    result[likelihood]['effects'] = prepare_survival_effects(parametric, X, times=[0, 1, 3])
    result[likelihood]['curve_interval'] = prepare_survival_curve_interval(parametric, X, times=[0, 1, 3], n_draws=64, seed=42)
    result[likelihood]['simultaneous_band'] = prepare_survival_curve_interval(parametric, X, times=[0, 1, 3], n_draws=64, seed=42, interval='simultaneous')
    result[likelihood]['effect_intervals'] = prepare_survival_effect_interval(parametric, X, times=[0, 1, 3], n_draws=64, seed=42, interval='simultaneous')
    result[likelihood]['hazard_interval'] = prepare_survival_hazard_interval(parametric, X, times=[.5, 1, 3], n_draws=64, seed=42, interval='simultaneous')
    # Small Monte Carlo budget/cohort keep this integration example quick.
    result[likelihood]['rmst_interval'] = prepare_survival_rmst_interval(parametric, X.iloc[:3], horizons=[0, 1, 3], conditional_on=target.entry[:3], n_draws=8, seed=42, interval='simultaneous')
    # Automatic joint REML uses the same observation and prediction contracts.
    rng = np.random.default_rng(286)
    x = rng.uniform(-1., 1., 180)
    location = .3 + .65*np.sin(np.pi*x)
    second = .12*np.cos(np.pi*x)
    time = (np.exp(location)*rng.weibull(2.1*np.exp(second)) if likelihood == 'weibull'
            else np.exp(location + .5*np.exp(second)*rng.normal(size=len(x))))
    entry = rng.uniform(.03, .25, len(x))
    entry[::5] = 0.
    kept = time > entry
    x, time, entry = x[kept], time[kept], entry[kept]
    censor = rng.uniform(1.2, 2.8, len(x))
    features = pd.DataFrame({'x': x})
    observed = SurvivalTarget(np.minimum(time, censor), time <= censor, entry)
    names = ('scale', 'shape') if likelihood == 'weibull' else ('loc', 'scale')
    automatic = SurvivalGAM(likelihood=likelihood,
        formula={names[0]: "~s(x,bs='cr',k=6)", names[1]: '~1'}).fit(features, observed)
    result[likelihood]['automatic_survival'] = automatic.predict_survival(features, times=[0, .5, 1, 2, 4])
    result[likelihood]['smoothing_parameters'] = np.asarray(automatic.gam_.smoothing_params)
