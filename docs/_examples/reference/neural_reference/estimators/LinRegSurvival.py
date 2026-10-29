"""Runnable reference example: neural_reference/estimators/LinRegSurvival."""

import numpy as np
import pandas as pd
from nampy.models import LinRegSurvival
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
from nampy import GradientTrainingConfig, SurvivalObjectiveConfig
model = LinRegSurvival(objective=SurvivalObjectiveConfig(cuts=(0, 1, 3)), training=GradientTrainingConfig(max_epochs=2, batch_size=16, trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False}))
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
# Independent replication is supplied explicitly, with its own fit seed.
from dataclasses import replace
replica = LinRegSurvival(objective=model.objective, training=replace(model.training, random_state=23)).fit(X, target)
result['between_fits'] = combine_survival_predictions(
    [prediction, replica.predict_distribution(X)], times=[0, .5, 1, 2, 3])
try:
    prepare_survival_effect_interval(model, X, times=[0, 1, 2])
except NotImplementedError:
    result['effect_coefficient_route'] = 'unsupported_single_fit_neural'
for operation, grid in (
        (prepare_survival_hazard_interval, {'times': [.5, 1]}),
        (prepare_survival_rmst_interval, {'horizons': [.5, 1]})):
    try:
        operation(model, X, **grid)
    except NotImplementedError:
        result[operation.__name__] = 'unsupported_single_fit_neural'
# Bootstrap refits need a fixed external validation cohort, not an automatic
# split of duplicate subject occurrences. These IDs identify original people.
bootstrap_target = SurvivalTarget(target.time, target.event, target.entry,
    subject_id=[f'person-{i}' for i in range(len(target))])
bootstrap_model = LinRegSurvival(objective=model.objective, training=model.training).fit(
    X.iloc[:25], bootstrap_target.take(np.arange(25)),
    X_val=X.iloc[25:], y_val=bootstrap_target.take(np.arange(25, 30)))
bootstrap = bootstrap_survival(bootstrap_model, X.iloc[:25],
    bootstrap_target.take(np.arange(25)), X.iloc[:4],
    X_val=X.iloc[25:], y_val=bootstrap_target.take(np.arange(25, 30)),
    resample_indices=np.stack([np.arange(25), np.r_[np.arange(24), 23]]),
    effects={'times': [0, .5, 1, 2, 3], 'reference_X': X.iloc[::2]})
result['subject_bootstrap'] = bootstrap.percentile_interval(grid=[0, .5, 1, 2, 3])

# Two prescribed refits demonstrate the API; use a substantial resampling budget
# for inference. Coverage here is joint only over the declared finite grid.
result['bootstrap_band'] = bootstrap.simultaneous_interval(grid=[0, .5, 1, 2, 3])
result['bootstrap_effects'] = bootstrap.effect_interval(interval='simultaneous')
# The same observation contract supports censored parametric time laws.
from tempfile import TemporaryDirectory
from pathlib import Path
for likelihood in ('weibull', 'lognormal'):
    parametric = LinRegSurvival(objective=SurvivalObjectiveConfig(likelihood=likelihood), training=model.training)
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
