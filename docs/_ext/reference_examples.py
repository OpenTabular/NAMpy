"""Standalone, executable examples for each public reference feature.

The example runner executes the exact files embedded in the documentation and
records output schemas. No examples import this generator at runtime.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from example_source import without_assertions

BASE = """import numpy as np
import pandas as pd
from nampy.gam import GAM

rng = np.random.default_rng(7)
n = 96
data = pd.DataFrame({'x': rng.uniform(0, 1, n), 'z': rng.uniform(0, 1, n)})
data['y'] = 0.4 + np.sin(3 * data.x) + 0.3 * data.z + rng.normal(0, 0.12, n)
"""
MODEL = """model = GAM(formula="y ~ s(x, bs='ps', k=7) + s(z, bs='cr', k=6)",
            optimize_smoothing=True, smoothing_method='reml').fit(data=data)
"""


def smooth_example(name, cls):
    code = BASE
    extra = ""
    if cls.__name__ == "BivariateShapePSplineTerm":
        return (
            code
            + f"""from nampy.gam.smooths import make_smooth_term

# Build the term's identified basis before choosing a coefficient optimizer.
term = make_smooth_term({name!r}, feature=['x', 'z'], k=[5, 6], m=[2, 1])
term.fit(data[['x', 'z']].to_numpy(), ['x', 'z'])
basis = term.basis_train
penalties = term.penalties
coefficients = np.linspace(0.05, 0.2, basis.shape[1])
surface = basis @ coefficients
assert surface.shape == (n,)
assert all(matrix.shape == (basis.shape[1], basis.shape[1]) for matrix in penalties)
result = {{'basis': basis, 'penalties': penalties, 'surface': surface}}
print(basis.shape, [matrix.shape for matrix in penalties])
"""
        )
    if name in {"mdcv", "mdcvby"}:
        code += "data['y'] = 2 - data.x - data.x**2 + rng.normal(0, 0.01, n)\n"
    args = f"x, bs='{name}', k=10"
    if name in {"te", "ti", "t2"}:
        args = "x, z, k=c(6, 5), bs=c('cr', 'cr')"
        formula = f"y ~ {name}({args})"
    else:
        if cls.__name__ == "BivariateShapePSplineTerm":
            args = f"x, z, bs='{name}', k=c(6, 5)"
        if name in {"ds", "ad"}:
            args = f"x, bs='{name}', k=12"
        if name in {"lmpi", "lipl"}:
            args += ", xt={'xc': 0.5}"
        if name.endswith("by"):
            code += "data['weight'] = rng.uniform(0.5, 1.5, n)\n"
            args += ", by=weight"
        if name in {"fs", "sz", "re"}:
            code += "data['group'] = pd.Categorical(np.resize(['a', 'b', 'c'], n))\n"
            args = "group, bs='re'" if name == "re" else f"x, group, bs='{name}', k=6"
        if name == "mrf":
            code += "data['region'] = pd.Categorical(np.resize(['a', 'b', 'c'], n))\n"
            args = "region, bs='mrf', k=3, xt={'nb': {'a': ['b'], 'b': ['a', 'c'], 'c': ['b']}}"
        if name == "sos":
            code += "data['latitude'] = rng.uniform(-50, 50, n)\ndata['longitude'] = rng.uniform(-100, 100, n)\n"
            args = "latitude, longitude, bs='sos', k=12"
        if name in {"so", "sf", "sw"}:
            code += (
                "boundary = [{'x': [0., 1., 1., 0., 0.], 'z': [0., 0., 1., 1., 0.]}]\n"
            )
            args = f"x, z, bs='{name}', k=8, xt={{'bnd': {{boundary}}, 'nmax': 25}}"
            extra = ", knots={'x': [0.3, 0.7, 0.3, 0.7], 'z': [0.3, 0.3, 0.7, 0.7]}"
        formula = f"y ~ s({args})"
    if name in {"so", "sf", "sw"}:
        code += f"formula = \"y ~ s(x, z, bs='{name}', k=8, xt={{'bnd': \" + repr(boundary) + \", 'nmax': 25}})\"\n"
        formula_expr = "formula"
    else:
        formula_expr = repr(formula)
    code += f"model = GAM(formula={formula_expr}, optimize_smoothing=False{extra}).fit(data=data)\n"
    code += """prediction = model.predict(data)
design = model.lpmatrix(data)
covariance = model.vcov()
assert prediction.shape == (n,)
assert design.shape[0] == n
assert covariance.shape == (design.shape[1], design.shape[1])
assert np.isfinite(prediction).all()
result = {'prediction': prediction, 'full_model_design': design,
          'coefficient_covariance': covariance}
print({key: value.shape for key, value in result.items()})
"""
    return code


def statistical_family_example(name):
    code = (
        "\n".join(
            line for line in BASE.splitlines() if not line.startswith("data['y'] =")
        )
        + "\n"
    )
    family = repr(name)
    formula = "y ~ s(x, bs='cr', k=6)"
    setup = {
        "gaussian": "data['y'] = 1 + data.x + rng.normal(0, 0.3, n)",
        "binomial": "data['y'] = rng.binomial(1, 1 / (1 + np.exp(-2 * (data.x - 0.5))))",
        "poisson": "data['y'] = rng.poisson(np.exp(0.5 + data.x))",
        "gamma": "data['y'] = rng.gamma(3, np.exp(0.3 * data.x) / 3)",
        "inverse_gaussian": "data['y'] = rng.wald(np.exp(0.3 * data.x), 2)",
        "nb": "data['y'] = rng.negative_binomial(2, 2 / (2 + np.exp(data.x)))",
        "negbin": "data['y'] = rng.negative_binomial(2, 2 / (2 + np.exp(data.x)))",
        "tw": "data['y'] = rng.gamma(2, 0.5, n) * (rng.random(n) > 0.2)",
        "betar": "data['y'] = rng.beta(2 + data.x, 3)",
        "ocat": "data['y'] = np.resize([1, 2, 3], n)",
        "scat": "data['y'] = data.x + 0.3 * rng.standard_t(5, n)",
        "zip": "data['y'] = rng.poisson(3, n) * (rng.random(n) > 0.3)",
        "cnorm": "data['y'] = 2 + data.x + rng.normal(0, 0.3, n)\ndata['bound'] = data.y.astype(float)\ndata.loc[:5, 'bound'] = np.inf",
        "clog": "data['y'] = 2 + data.x + rng.logistic(0, 0.3, n)\ndata['bound'] = data.y.astype(float)\ndata.loc[:5, 'bound'] = np.inf",
        "bcg": "data['y'] = np.exp(0.3 + 0.2 * data.x + rng.normal(0, 0.15, n))",
        "cpois": "data['y'] = 1 + rng.poisson(np.exp(0.5 + data.x))\ndata['bound'] = data.y.astype(float)\ndata.loc[:5, 'bound'] = np.inf",
        "quasi": "data['y'] = rng.wald((1 + 0.3 * data.x)**2, 200.0)",
        "quasipoisson": "data['y'] = rng.poisson(np.exp(0.5 + data.x))",
        "quasibinomial": "data['y'] = rng.binomial(1, 0.2 + 0.5 * data.x)",
        "gaulss": "data['y'] = data.x + rng.normal(0, 0.2 + 0.3 * data.x, n)",
        "gammals": "data['y'] = rng.gamma(3, np.exp(0.3 * data.x) / 3)",
        "gumbls": "data['y'] = rng.gumbel(data.x, 0.3, n)",
        "ziplss": "data['y'] = rng.poisson(3, n) * (rng.random(n) > 0.3)",
        "multinom": "data['y'] = np.resize([0, 1, 2], n)",
        "gevlss": "data['y'] = rng.gumbel(data.x, 0.3, n)",
        "shash": "data['y'] = data.x + rng.normal(0, 0.3, n)",
        "twlss": "data['y'] = rng.gamma(2, 0.5, n) * (rng.random(n) > 0.2)",
    }
    code += setup[name] + "\n"
    if name == "negbin":
        family = "{'name': 'negbin', 'theta': 2.0}"
    if name == "nb":
        family = "{'name': 'nb', 'theta': 2.0, 'estimate_theta': False}"
    if name == "ocat":
        family = "{'name': 'ocat', 'R': 3}"
    if name in {"cnorm", "clog"}:
        family = f"{{'name': '{name}', 'theta': 0.3}}"
        formula = "cbind(y, bound) ~ s(x, bs='cr', k=6)"
    if name == "cpois":
        formula = "cbind(y, bound) ~ s(x, bs='cr', k=6)"
    if name == "bcg":
        code += "from nampy.gam.families.registry import make_gam_family\nfamily = make_gam_family({'name': 'bcg'})\nfamily.putTheta([0.0, np.log(0.3)])\nfamily.n_theta = 0\n"
        family = "family"
    if name == "quasi":
        family = "{'name': 'quasi', 'variance': 'mu^3', 'power': 0.5}"
    from nampy.gam.families.registry import GENERAL_FAMILIES, make_gam_family

    if name in GENERAL_FAMILIES:
        if name == "multinom":
            family = "{'name': 'multinom', 'K': 2}"
        f = make_gam_family({"name": name, "K": 2} if name == "multinom" else name)
        count = f.n_linear_predictors
        formulas = [formula] + ["~ 1"] * (count - 1)
        code += f"model = GAM(formula={formulas!r}, family={family}, optimize_smoothing=False).fit(data=data)\n"
        code += """raw = model.predict(data, type='link')
parameters = model.family.distribution_parameters_from_eta(np.asarray(raw))
source_response = np.asarray(model.predict(data, type='response'))
result = {'link_predictors': np.asarray(raw), 'natural_parameters': parameters, 'source_response': source_response}
assert np.isfinite(source_response).all()
assert parameters.shape[0] == n
assert np.isfinite(parameters).all()
print({key: value.shape for key, value in result.items()})
"""
    else:
        code += f"model = GAM(formula={formula!r}, family={family}, optimize_smoothing=False).fit(data=data)\n"
        code += """prediction = model.predict(data, type='response')
raw = model.predict(data, type='link')
assert np.asarray(prediction).shape[0] == n
assert np.isfinite(prediction).all()
result = {'response_prediction': np.asarray(prediction), 'link_prediction': np.asarray(raw)}
print({key: value.shape for key, value in result.items()})
"""
    return code


def method_example(name):
    code = BASE
    if name == "ar1_standardized_residuals":
        code += MODEL.replace(
            "optimize_smoothing=True", "ar1_rho=0.3, optimize_smoothing=False"
        )
    else:
        code += MODEL
    calls = {
        "fit": "model",
        "fisher_information": "model.fisher_information(data.iloc[:4])",
        "predict": "model.predict(data, return_se=True)",
        "predict_terms": "model.predict_terms(data)",
        "lpmatrix": "model.lpmatrix(data)",
        "summary": "model.summary()",
        "residuals": "model.residuals(type='deviance')",
        "ar1_standardized_residuals": "model.ar1_standardized_residuals()",
        "concurvity": "model.concurvity(full=False)",
        "k_check": "model.k_check(n_rep=10, seed=7)",
        "gam_check": "model.gam_check(k_rep=10, seed=7)",
        "derivative": "model.derivative(smooth_number=1, deriv=1)",
        "vcov": "model.vcov(unconditional=True)",
        "sp_vcov": "model.sp_vcov()",
        "gam_vcomp": "model.gam_vcomp()",
        "one_se_rule": "{'original_smoothing': np.asarray(model.smoothing_params).copy(), 'proposed_smoothing': model.one_se_rule()}",
        "anova": "model.anova()",
        "aic": "model.aic()",
        "bic": "model.bic()",
        "loglik": "model.loglik()",
        "edf1": "model.edf1()",
        "fit_result": "model.fit_result(include_covariances=True)",
        "parity_snapshot": "model.parity_snapshot(data)",
        "get_device": "model.get_device()",
        "plot": "model.plot(n=30, pages=1)",
        "qq": "{'analytic': model.qq(rep=0, s_rep=3, seed=7, quantile_permutations=permutations), 'simulated': model.qq(rep=3, seed=7, response_draws=draws)}",
        "plot_check": "{'analytic': model.plot_check(rep=0, s_rep=3, seed=7, quantile_permutations=permutations), 'simulated': model.plot_check(rep=3, seed=7, response_draws=draws)}",
        "vis": "model.vis(view=['x', 'z'], n_grid=8, plot_type='contour')",
        "plot_derivative": "model.plot_derivative(smooth_number=1, deriv=1)",
    }
    if name in {"qq", "plot_check"}:
        code += "permutations = np.vstack([rng.permutation(n) for _ in range(3)])\ndraws = model.predict(data)[None, :] + rng.normal(0, np.sqrt(model.fit_result().scale), size=(3, n))\nassert permutations.shape == draws.shape == (3, n)\n"
    if name in {"save_model", "load_model"}:
        code += """from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / 'fitted.nampy'
    saved = model.save_model(path)
    restored = GAM.load_model(path)
    assert path.is_file()
    np.testing.assert_allclose(restored.predict(data), model.predict(data))
"""
        code += f"    result = {'saved' if name == 'save_model' else 'restored'}\n"
    elif name == "predict":
        code += "prediction, standard_error = model.predict(data, return_se=True)\nresult = {'prediction': prediction[:3], 'standard_error': standard_error[:3]}\n"
    else:
        code += f"result = {calls[name]}\n"
    code += (
        "print(result)\n"
        if name in {"predict", "one_se_rule"}
        else "print(type(result).__name__)\n"
    )
    return code


def selection_example(name):
    from feature_reference import OPTIMIZER_MEANINGS

    option = "smoothing_optimizer" if name in OPTIMIZER_MEANINGS else "smoothing_method"
    kwargs = f"{option}={name!r}, optimize_smoothing={name != 'fixed'}"
    if option == "smoothing_optimizer":
        kwargs += ", smoothing_method='reml'"
    if name == "ubre":
        kwargs += ", scale=1.0"
    return (
        BASE
        + f"model = GAM(formula=\"y ~ s(x, bs='cr', k=6)\", {kwargs}).fit(data=data)\nresult = np.asarray(model.smoothing_params)\nassert result.ndim == 1\nassert np.isfinite(model.predict(data)).all()\nprint(result.shape)\n"
    )


def neural_family_example(name):
    from nampy.neural.families.registry import FAMILY_REGISTRY

    spec = FAMILY_REGISTRY[name]
    cls = spec.distribution
    kwargs = {"trials":10} if name=="betabinomial" else {}
    if name in {"lognormal_observation", "weibull_observation"}:
        kwargs.update(lower=0.2, upper=3.0)
    if name in {"categorical", "ordinal"}:
        kwargs["num_classes"] = 3
    if name in {"dirichlet", "mvnormdiag"}:
        kwargs["n_dim"] = 3
    kwargs_text = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    code = f"import numpy as np\nimport torch\nfrom {cls.__module__} import {cls.__name__}\n\nfamily = {cls.__name__}({kwargs_text})\nn = 8\nraw = torch.zeros((n, family.param_count), dtype=torch.float32)\n"
    if name in {"bct", "bcpe"}:
        code += f"raw[:, 1] = -1\nraw[:, 2] = 1\nraw[:, 3] = np.log({5 if name == 'bct' else 2})\n"
        target = "torch.linspace(1.1, 1.8, n)"
    elif name in {"lognormal_observation", "weibull_observation"}:
        code += "from nampy.distributions import ObservationTarget\nobservation = ObservationTarget([0.6, -np.inf, 1.5, 0.7] * 2, [0.6, 0.5, np.inf, 1.1] * 2)\n"
        target = "family.encode_targets(observation)"
    elif name == "normal_observation":
        target = "torch.column_stack([torch.ones(n),torch.ones(n),torch.zeros(n)])"
    elif name in {"categorical", "ordinal"}:
        target = "torch.arange(n) % 3"
    elif name == "dirichlet":
        target = "torch.full((n, 3), 1 / 3)"
    elif name == "mvnormdiag":
        target = "torch.ones((n, 3))"
    elif name == "beta":
        target = "torch.full((n,), 0.4)"
    else:
        target = "torch.ones(n)"
    code += f"target = {target}\n"
    code += """parameters = family(raw)
loss = family.compute_loss(raw, target)
point = family.predict_point(raw)
assert parameters.shape[0] == n
assert loss.ndim == 0 and torch.isfinite(loss)
result = {'raw': raw, 'parameters': parameters, 'loss': loss, 'point': point}
print({key: tuple(value.shape) for key, value in result.items()})
"""
    return code


def neural_estimator_example(name, spec=None, task=None):
    if task == "survival" or name.endswith("Survival"):
        return survival_estimator_example(name)
    from nampy import models

    cls = getattr(models, name)
    defaults = vars(cls().architecture_config)
    options = {}
    for key, value in {
        "layer_sizes": [12, 8],
        "reference_epochs": 1,
        "reference_layer_sizes": [12, 8],
        "n_estimators": 2,
        "num_estimators": 2,
        "num_trees": 3,
        "tree_depth": 2,
        "depth": 2,
        "num_bases": 8,
        "rff_num_feat": 16,
        "n_knots": 6,
    }.items():
        if key in defaults:
            options[key] = value
    if "interactions" in defaults:
        options["interactions"] = []
    if name.startswith("NodeGAM"):
        options.update(
            num_layers=1, anneal_steps=1, last_dropout=0.0, interaction_degree=1
        )
    if name.endswith("LSS"):
        task = "distributional"
    elif name.endswith("Classifier"):
        task = "classification"
    else:
        task = "regression"
    code = (
        """import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import torch
"""
        + f"from nampy.models import {name}\n\n"
        + """torch.set_num_threads(1)
rng = np.random.default_rng(7)
n = 64
X = pd.DataFrame({'x': np.linspace(-1, 1, n), 'z': rng.uniform(-1, 1, n)})
"""
    )
    code += (
        "y = (np.arange(n) % 2).astype(int)\n"
        if task == "classification"
        else "y = np.sin(3 * X.x.to_numpy()) + rng.normal(0, 0.1, n)\n"
    )
    from dataclasses import fields

    from nampy.neural.configuration import GradientTrainingConfig

    estimator = cls()
    config_class = type(estimator.architecture_config)
    route = estimator._resolved_training([0, 1])
    route_class = type(route).__name__
    code += f"from {config_class.__module__} import {config_class.__name__}\n"
    code += f"from nampy.neural.configuration import {route_class}\n"
    arguments = ", ".join(f"{key}={value!r}" for key, value in options.items())
    controls = "batch_size=32, random_state=7"
    baseline = type(route)()
    for field in fields(route):
        if field.name not in {"batch_size", "random_state"} and getattr(route, field.name) != getattr(baseline, field.name):
            controls += f", {field.name}={getattr(route, field.name)!r}"
    if isinstance(route, GradientTrainingConfig):
        controls += ", max_epochs=2, trainer_kwargs={'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False}"
    code += f"model = {name}(architecture_config={config_class.__name__}({arguments}), training={route_class}({controls}))\n"
    code += "with TemporaryDirectory() as directory:\n"
    if isinstance(route, GradientTrainingConfig):
        code += "    model.set_params(training__checkpoint_path=directory)\n"
    code += "    model.fit(X, y)\n"
    code += """prediction = model.predict(X)
assert np.asarray(prediction).shape[0] == n
assert np.isfinite(prediction).all()
result = {'prediction': prediction}
"""
    if task == "classification":
        code += """probability = model.predict_proba(X)
assert probability.shape == (n, 2)
np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-5)
result['probabilities'] = probability
"""
    elif task == "distributional":
        code += "result['raw'] = model.predict_link(X)\nresult['point'] = model.predict_point(X)\n"
    if spec is None or spec.supports("additive_components"):
        code += "components = model.predict_components(X)\ncomponents.validate_additive_reconstruction(rtol=1e-4, atol=1e-5)\nresult['components'] = components\n"
    code += "print({key: type(value).__name__ for key, value in result.items()})\n"
    return code


def statistical_estimator_example(name):
    if name == "CalibratedQuantileGAM":
        return BASE.replace("rng.normal(0, 0.12, n)", "rng.normal(0, 0.4, n)") + """from nampy import CalibratedQuantileGAM
model = CalibratedQuantileGAM(formula='~s(x,k=4)+z', level=.3,
    bandwidth=.2, smoothing_params=[.5], log_scale_bounds=(-3,2)).fit(data[['x','z']], data.y)
interval = model.predict_interval(data[['x','z']])
result = {'prediction': model.predict(data[['x','z']]),
          'interval': interval, 'components': model.predict_components(data[['x','z']])}
print(interval.coverage, model.loss_scale_)
"""
    if name == "CoxGAM":
        return BASE + """from nampy import CoxGAM
from nampy.survival import SurvivalTarget
record = np.arange(n)
target = SurvivalTarget(.2+((record*13)%27)/10, record%4!=1)
model = CoxGAM(formula='~x+z', tol=1e-8).fit(data[['x','z']], target)
prediction = model.predict_distribution(data[['x','z']].iloc[:4])
result = {'survival': prediction.survival([.5,1,2]),
          'rmst': prediction.rmst(2),
          'components': model.predict_components(data[['x','z']].iloc[:4])}
print({key: type(value).__name__ for key,value in result.items()})
"""
    if name == "SurvivalGAM":
        return survival_estimator_example(name)
    code = BASE + "from nampy.models import " + name + "\n"
    if name == "GAMLSS":
        code += "model = GAMLSS(family='gaulss', optimize_smoothing=False, k=6).fit(data[['x', 'z']], data.y)\nresult = {'natural_parameters': model.predict_parameters(data[['x', 'z']]), 'raw': model.predict_link(data[['x', 'z']]), 'point': model.predict_point(data[['x', 'z']])}\n"
    else:
        if name == "GAMClassifier":
            code += "data['y'] = (np.arange(n) % 2).astype(int)\n"
        code += f"model = {name}(optimize_smoothing=False, k=6).fit(data[['x', 'z']], data.y)\nresult = {{'prediction': model.predict(data[['x', 'z']]), 'components': model.predict_components(data[['x', 'z']])}}\n"
        if name == "GAMClassifier":
            code += "result['probabilities'] = model.predict_proba(data[['x', 'z']])\n"
    code += "print({key: type(value).__name__ for key, value in result.items()})\n"
    return code


def survival_estimator_example(name):
    code = f"""import numpy as np
import pandas as pd
from nampy.models import {name}
from nampy.survival import (SurvivalTarget, CensoringDistribution, brier_score,
                            concordance_index, survival_curve,
                            prepare_survival_curve_interval,
                            prepare_survival_effect_interval, combine_survival_predictions,
                            prepare_survival_hazard_interval, prepare_survival_rmst_interval,
                            bootstrap_survival)
from nampy.diagnostics import survival_residuals, prepare_survival_effects

X = pd.DataFrame({{'x': np.linspace(-1., 1., 30)}})
target = SurvivalTarget(
    time=np.r_[np.linspace(.1, .9, 15), np.linspace(1.1, 2.9, 15)],
    event=np.tile([1, 0, 1], 10), entry=np.full(30, .02),
)
"""
    if name == "SurvivalGAM":
        code += "model = SurvivalGAM(cuts=[0, 1, 3], formula='~ x', optimize_smoothing=False)\n"
    else:
        code += "from nampy import GradientTrainingConfig, SurvivalObjectiveConfig\n"
        extra = ""
        if name == "NAMSurvival":
            code += "from nampy.neural.configs.nam_config import DefaultNAMConfig\n"
            extra = "architecture_config=DefaultNAMConfig(layer_sizes=[4], dropout=0), "
        code += f"model = {name}({extra}objective=SurvivalObjectiveConfig(cuts=(0, 1, 3)), training=GradientTrainingConfig(max_epochs=2, batch_size=16, trainer_kwargs={{'logger': False, 'enable_progress_bar': False, 'enable_model_summary': False}}))\n"
    code += """model.fit(X, target)
prediction = model.predict_distribution(X)
result = {'survival': prediction.survival([0, .5, 1, 2, 3]),
          'hazard': prediction.hazard([.5, 2]), 'rmst': prediction.rmst(3),
          'components': model.predict_components(X), 'log_likelihood': model.score(X, target)}
assert result['survival'].shape == (30, 5)
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
"""
    if name == "SurvivalGAM":
        code += "result['curve_interval'] = prepare_survival_curve_interval(model, X, times=[0, .5, 1, 2, 3], n_draws=64, seed=42)\n"
        code += "result['simultaneous_band'] = prepare_survival_curve_interval(model, X, times=[0, .5, 1, 2, 3], n_draws=64, seed=42, interval='simultaneous')\n"
        code += "result['effect_intervals'] = prepare_survival_effect_interval(model, X, times=[0, .5, 1, 2, 3], reference_X=X.iloc[::2], n_draws=64, seed=42, interval='simultaneous')\n"
        code += "result['hazard_interval'] = prepare_survival_hazard_interval(model, X, times=[.5, 1, 2, 3], n_draws=64, seed=42, interval='simultaneous')\n"
        code += "result['rmst_interval'] = prepare_survival_rmst_interval(model, X, horizons=[0, 1, 2, 3], n_draws=64, seed=42, interval='simultaneous')\n"
        code += "bootstrap = bootstrap_survival(model, X, target, X.iloc[:4], resample_indices=np.stack([np.arange(30), np.r_[np.arange(29), 28]]), effects={'times': [0, .5, 1, 2, 3], 'reference_X': X.iloc[::2]})\n"
        code += "result['subject_bootstrap'] = bootstrap.percentile_interval(grid=[0, .5, 1, 2, 3])\n"
    else:
        code += f"""# Independent replication is supplied explicitly, with its own fit seed.
from dataclasses import replace
replica = {name}({extra}objective=model.objective, training=replace(model.training, random_state=23)).fit(X, target)
result['between_fits'] = combine_survival_predictions(
    [prediction, replica.predict_distribution(X)], times=[0, .5, 1, 2, 3])
try:
    prepare_survival_effect_interval(model, X, times=[0, 1, 2])
except NotImplementedError:
    result['effect_coefficient_route'] = 'unsupported_single_fit_neural'
for operation, grid in (
        (prepare_survival_hazard_interval, {{'times': [.5, 1]}}),
        (prepare_survival_rmst_interval, {{'horizons': [.5, 1]}})):
    try:
        operation(model, X, **grid)
    except NotImplementedError:
        result[operation.__name__] = 'unsupported_single_fit_neural'
# Bootstrap refits need a fixed external validation cohort, not an automatic
# split of duplicate subject occurrences. These IDs identify original people.
bootstrap_target = SurvivalTarget(target.time, target.event, target.entry,
    subject_id=[f'person-{{i}}' for i in range(len(target))])
bootstrap_model = {name}({extra}objective=model.objective, training=model.training).fit(
    X.iloc[:25], bootstrap_target.take(np.arange(25)),
    X_val=X.iloc[25:], y_val=bootstrap_target.take(np.arange(25, 30)))
bootstrap = bootstrap_survival(bootstrap_model, X.iloc[:25],
    bootstrap_target.take(np.arange(25)), X.iloc[:4],
    X_val=X.iloc[25:], y_val=bootstrap_target.take(np.arange(25, 30)),
    resample_indices=np.stack([np.arange(25), np.r_[np.arange(24), 23]]),
    effects={{'times': [0, .5, 1, 2, 3], 'reference_X': X.iloc[::2]}})
result['subject_bootstrap'] = bootstrap.percentile_interval(grid=[0, .5, 1, 2, 3])
"""
    code += """
# Two prescribed refits demonstrate the API; use a substantial resampling budget
# for inference. Coverage here is joint only over the declared finite grid.
result['bootstrap_band'] = bootstrap.simultaneous_interval(grid=[0, .5, 1, 2, 3])
result['bootstrap_effects'] = bootstrap.effect_interval(interval='simultaneous')
# The same observation contract supports censored parametric time laws.
from tempfile import TemporaryDirectory
from pathlib import Path
for likelihood in ('weibull', 'lognormal'):
"""
    if name == "SurvivalGAM":
        code += "    parametric = SurvivalGAM(likelihood=likelihood, formula='~ x', optimize_smoothing=False)\n"
    else:
        code += f"    parametric = {name}({extra}objective=SurvivalObjectiveConfig(likelihood=likelihood), training=model.training)\n"
    code += """    parametric.fit(X, target)
    latent = parametric.predict_distribution(X)
    with TemporaryDirectory() as directory:
        saved = parametric.save_model(Path(directory) / 'survival.nampy')
        restored = type(parametric).load_model(saved)
        assert np.array_equal(restored.predict_survival(X, times=[0, 1, 3]),
                              latent.survival([0, 1, 3]))
    result[likelihood] = {'survival': latent.survival([0, 1, 3]),
                          'parameters': dict(latent.distribution.parameters),
                          'rmst': latent.rmst(3),
                          'remaining_rmst': latent.rmst(3, conditional_on=target.entry),
                          'residuals': survival_residuals(latent, target),
                          'log_likelihood': parametric.score(X, target)}
"""
    code += "    result[likelihood]['effects'] = prepare_survival_effects(parametric, X, times=[0, 1, 3])\n"
    if name == "SurvivalGAM":
        code += "    result[likelihood]['curve_interval'] = prepare_survival_curve_interval(parametric, X, times=[0, 1, 3], n_draws=64, seed=42)\n"
        code += "    result[likelihood]['simultaneous_band'] = prepare_survival_curve_interval(parametric, X, times=[0, 1, 3], n_draws=64, seed=42, interval='simultaneous')\n"
        code += "    result[likelihood]['effect_intervals'] = prepare_survival_effect_interval(parametric, X, times=[0, 1, 3], n_draws=64, seed=42, interval='simultaneous')\n"
        code += "    result[likelihood]['hazard_interval'] = prepare_survival_hazard_interval(parametric, X, times=[.5, 1, 3], n_draws=64, seed=42, interval='simultaneous')\n"
        code += "    # Small Monte Carlo budget/cohort keep this integration example quick.\n"
        code += "    result[likelihood]['rmst_interval'] = prepare_survival_rmst_interval(parametric, X.iloc[:3], horizons=[0, 1, 3], conditional_on=target.entry[:3], n_draws=8, seed=42, interval='simultaneous')\n"
    if name == "SurvivalGAM":
        code += """    # Automatic joint REML uses the same observation and prediction contracts.
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
    assert automatic.parametric_.distribution_fit_status().eligible
"""
    return code


def control_example(name):
    from public_reference import literal

    from nampy.gam import gam_control

    value = getattr(gam_control(), name)
    return f"from nampy.gam import gam_control\n\ncontrol = gam_control({name}={literal(value)})\nresult = {{{name!r}: control.{name}}}\nassert hasattr(control, {name!r})\nprint(control.{name})\n"


def extension_example(name):
    if name == "NeuralEnsemble":
        code = neural_estimator_example("NAMRegressor")
        code = code.replace(
            "from nampy.models import NAMRegressor",
            "from nampy.models import NAMRegressor, NeuralEnsemble",
        )
        code = code.replace("model = NAMRegressor(", "base = NAMRegressor(")
        code = code.replace(
            "with TemporaryDirectory() as directory:",
            "model = NeuralEnsemble(base, n_estimators=2, random_state=7)\nwith TemporaryDirectory() as directory:",
        )
        code = code.replace("model.set_params(training__checkpoint_path=directory)",
                            "model.estimator.set_params(training__checkpoint_path=directory)")
        code += (
            "result['component_uncertainty'] = model.predict_component_uncertainty(X)\n"
        )
        return code
    if name in {"estimator_family", "NeuralEstimatorFamily"}:
        return """from nampy.models import estimator_family, NeuralEstimatorFamily

registered = estimator_family('nam', module_name=__name__)
result = NeuralEstimatorFamily(regressor=registered.regressor,
                              classifier=registered.classifier, lss=registered.lss)
assert result.regressor is not None
print(result)
"""
    task = {
        "NeuralRegressor": "NAMRegressor",
        "NeuralClassifier": "NAMClassifier",
        "NeuralLSS": "NAMLSS",
    }[name]
    code = neural_estimator_example(task)
    code = code.replace(
        f"from nampy.models import {task}",
        f"from nampy.models import {name}\nfrom nampy.neural.architectures import NAM\nfrom nampy.neural.configs import DefaultNAMConfig",
    )
    code = code.replace(f"model = {task}(", f"model = {name}(model=NAM, config_class=DefaultNAMConfig, ")
    return code


def example_rst(root, page, code):
    """Write exact downloadable source and embed it on the feature page."""
    path = Path(root) / "_examples/reference" / f"{page}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        '"""Runnable reference example: '
        + page
        + '."""\n\n'
        + without_assertions(textwrap.dedent(code)).strip()
        + "\n"
    )
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    relative = "/_examples/reference/" + page + ".py"
    result = "\n\nRunnable example\n----------------\n\n"
    result += (
        "This example includes its data and imports. Run it after installing NAMpy. Small neural training budgets demonstrate the API; they do not establish predictive quality.\n\n"
        if page.startswith("neural_reference/estimators")
        else "This example includes all imports and input data; no variables from another page are required.\n\n"
    )
    result += f":download:`Download the complete example <{relative}>`\n\n.. literalinclude:: {relative}\n   :language: python\n\n"
    preview_path = Path(root) / "_examples/reference/output_previews.json"
    schema_path = Path(root) / "_examples/reference/output_schemas.json"
    if preview_path.exists() and schema_path.exists():
        from example_results import result_rst

        previews = json.loads(preview_path.read_text())
        schemas = json.loads(schema_path.read_text())
        if page in previews:
            preview = previews[page]
            if "/families/" in page and isinstance(preview, dict):
                from copy import deepcopy

                preview = deepcopy(preview)
                name = page.rsplit("/", 1)[-1]
                if page.startswith("statistical_reference/"):
                    from nampy.gam.families.registry import make_gam_family

                    family = make_gam_family(
                        {
                            "name": name,
                            **(
                                {"R": 3}
                                if name == "ocat"
                                else {"theta": 2.0}
                                if name == "negbin"
                                else {}
                            ),
                        }
                    )
                    columns = getattr(
                        family, "distribution_parameter_names", None
                    ) or getattr(family, "parameter_names", ())
                else:
                    import numpy as np

                    from nampy.neural.families.registry import FAMILY_REGISTRY

                    # Match the standalone example's inferred target width.
                    family, _ = FAMILY_REGISTRY[name].instantiate(
                        np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4]])
                        if name == "dirichlet"
                        else np.zeros((8, 3))
                        if name == "mvnormdiag"
                        else np.array([0, 1, 2])
                        if name in {"categorical", "ordinal"}
                        else np.zeros(8),
                        {"trials":10} if name=="betabinomial" else {},
                    )
                    columns = (
                        [f"p{i}" for i in range(family.num_classes)]
                        if name == "ordinal"
                        else ["mu","phi","pi0","pi1","pi_interior"] if name=="beinf" else family.param_names
                    )
                for key in ("parameters", "natural", "natural_parameters"):
                    value = preview.get(key)
                    if (
                        isinstance(value, dict)
                        and len(value.get("shape", [])) == 2
                        and value["shape"][1] == len(columns)
                    ):
                        value["columns"] = list(columns)
            result += result_rst(preview, schemas.get(page, []))
    return result
