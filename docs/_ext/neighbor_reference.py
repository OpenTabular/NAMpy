"""Reference pages for asymmetric and structured statistical extensions."""

from pathlib import Path

from reference_examples import BASE, MODEL, example_rst

ASYMMETRIC = """import numpy as np
import pandas as pd
from nampy import {name}

rng = np.random.default_rng(19)
x = np.linspace(-1, 1, 60)
data = pd.DataFrame({{'x': x, 'y': np.sin(3*x) + rng.normal(0, .25, len(x))}})
model = {name}(formula='y ~ s(x, k=6)', level=.8, smoothing_params=[.1]{extra}).fit(data)
prediction = model.predict(data)
components = model.predict_components(data)
figures = model.plot_terms(data)
assert prediction.shape == (60,)
assert model.fit_result_.converged
result = {{'prediction': prediction, 'components': components, 'fit': model.fit_result_}}
"""

LINEAR = """import numpy as np
from nampy.gam.losses.objectives import {name}

x = np.linspace(-1, 1, 40)
design = np.column_stack([np.ones(len(x)), x, x**2])
y = .4 + .5*x + .2*np.sin(np.arange(len(x)))
penalty = np.diag([0., .1, .2])
result = {call}
"""

SURVIVAL = """import numpy as np
import pandas as pd
from nampy import GAMRegressor
from nampy.survival import piecewise_exponential_data, survival_from_hazard

rng = np.random.default_rng(24)
x = rng.normal(size=100)
event_time = rng.exponential(scale=np.exp(-.3*x))
censor_time = rng.uniform(.5, 4, size=100)
subjects = pd.DataFrame({'x': x, 'time': np.minimum(event_time, censor_time),
                         'event': (event_time <= censor_time).astype(int)})
cuts = np.linspace(0, 4, 9)
ped = piecewise_exponential_data(subjects, time='time', event='event', cuts=cuts)
model = GAMRegressor(formula="ped_event ~ s(tend, bs='cr', k=5) + x", family='poisson',
                     optimize_smoothing=False, smoothing_params=[.2]).fit(ped, offset=ped.offset)
profile = pd.DataFrame({'tend': cuts[1:], 'x': np.zeros(8)})
hazard = model.predict(profile, offset=np.zeros(8))[None, :]
curves = survival_from_hazard(hazard, cuts)
assert np.all(np.diff(curves['survival'][0]) <= 0)
result = {'expanded': ped, 'hazard': hazard, **curves}
"""

LAGS = """import numpy as np
import pandas as pd
from nampy import GAMRegressor
from nampy.gam.structured.lags import exposure_history, cross_basis, cumulative_lag_matrix

rng = np.random.default_rng(42)
exposure = rng.normal(size=100)
history = exposure_history(exposure, max_lag=3)
lag = np.column_stack([np.ones(4), np.arange(4)])
def exposure_basis(x):
    return np.column_stack([x, x*x])
basis = cross_basis(history, exposure_basis=exposure_basis, lag_basis=lag)
complete = np.all(np.isfinite(basis), axis=1)
columns = ['b0','b1','b2','b3']
data = pd.DataFrame(basis[complete], columns=columns)
data['y'] = basis[complete] @ np.array([.2,-.03,.1,-.01]) + rng.normal(0,.2, complete.sum())
model = GAMRegressor(formula='y ~ b0+b1+b2+b3', optimize_smoothing=False).fit(data)
contrast = cumulative_lag_matrix([0.,1.,2.], 0., n_lags=4,
                                exposure_basis=exposure_basis, lag_basis=lag)
result = {'history': history, 'basis': basis, 'predictions': model.predict(data), 'cumulative_design': contrast}
"""

FUNCTIONS = {
    "DistributedLagBasis": ("Fitted distributed-lag bases", "import numpy as np\nfrom nampy.gam.structured import DistributedLagBasis\nhistory = np.random.default_rng(4).normal(size=(30,4))\nbasis = DistributedLagBasis(max_lag=3, exposure_formula='~exposure',lag_formula='~lag').fit(history)\nresult = {'matrix': basis.transform(history), 'penalty': basis.penalty(), 'contrast': basis.cumulative_contrast([0.,1.],0.)}\nassert result['matrix'].shape == (30,2)\n"),
    "fit_expectile": (
        "LAWS expectile solver",
        LINEAR.format(
            name="fit_expectile",
            call="fit_expectile(design, y, level=.8, penalty=penalty)",
        ),
    ),
    "fit_quantile": (
        "Penalized quantile solver",
        LINEAR.format(
            name="fit_quantile",
            call="{'exact': fit_quantile(design, y, level=.8, penalty=penalty), 'smooth': fit_quantile(design, y, level=.8, penalty=penalty, bandwidth=.1)}",
        ),
    ),
    "LossFit": (
        "Asymmetric fit result",
        LINEAR.format(
            name="fit_expectile", call="fit_expectile(design, y, penalty=penalty)"
        ),
    ),
    "coefficient_draws": (
        "Joint coefficient simulation",
        BASE
        + MODEL
        + "\nfrom nampy.gam.inference.functionals import coefficient_draws\nresult = coefficient_draws(model, n_draws=2000, seed=4)\nassert result.ndim == 2\n",
    ),
    "linear_functional_interval": (
        "Simultaneous bands and contrasts",
        BASE
        + MODEL
        + "\nfrom nampy.gam.inference.functionals import linear_functional_interval\nleft = data.copy()\nright = data.copy()\nleft['x'] = .2\nright['x'] = .8\ncontrast = model.lpmatrix(right) - model.lpmatrix(left)\nresult = linear_functional_interval(model, contrast, n_draws=2000, seed=4)\n",
    ),
    "derivative_interval": (
        "Derivative bands and significant regions",
        BASE
        + MODEL
        + "\nfrom nampy.gam.inference.functionals import derivative_interval\nresult = derivative_interval(model, data, 'x', n_draws=2000, seed=4)\nassert result.significant.shape == (n,)\n",
    ),
    "FunctionalInterval": (
        "Functional interval result",
        BASE
        + MODEL
        + "\nfrom nampy.gam.inference.functionals import linear_functional_interval\nresult = linear_functional_interval(model, model.lpmatrix(data), interval='pointwise')\n",
    ),
    "joint_edf": (
        "Joint effective degrees of freedom",
        "import numpy as np\nfrom nampy.gam.inference.functionals import joint_edf\ninformation = np.array([[4.,1.3],[1.3,2.]])\npenalty = np.diag([.2,3.])\nedf, total, edf1 = joint_edf(information, penalty)\nresult = {'edf': edf, 'total': total, 'edf1': edf1}\nassert total < 2\n",
    ),
    "piecewise_exponential_data": ("Piecewise exponential survival data", SURVIVAL),
    "survival_from_hazard": ("Hazard integration and survival prediction", SURVIVAL),
    "exposure_history": ("Exposure histories", LAGS),
    "cross_basis": ("Distributed-lag nonlinear cross-basis", LAGS),
    "cumulative_lag_matrix": ("Cumulative lag contrasts", LAGS),
    "cross_basis_penalty": (
        "Distributed-lag tensor penalties",
        "import numpy as np\nfrom nampy.gam.structured.lags import cross_basis_penalty\nresult = cross_basis_penalty(np.diag([0.,1.]), np.diag([0.,1.,1.]), exposure_weight=.1, lag_weight=.5)\nassert result.shape == (6,6)\n",
    ),
}

# Authored example data only; runtime classes/parameter names come from the registry.
FAMILY_RESPONSES = {
    "bct_lss": "np.exp(.3+.2*x)*DistributionPrediction(BCTLaw(), {'mu': [1.], 'sigma': [.7], 'nu': [1.2], 'tau': [4.]}).quantile(rng.permutation((np.arange(n)+.5)/n))[0]",
    "bcpe_lss": "np.exp(.3+.2*x)*DistributionPrediction(BCPELaw(), {'mu': [1.], 'sigma': [.7], 'nu': [1.2], 'tau': [1.4]}).quantile(rng.permutation((np.arange(n)+.5)/n))[0]",
    "lognormal_observation_lss": "np.exp(rng.normal(.2+.3*x,.4,n))",
    "weibull_observation_lss": "np.exp(.2+.3*x)*rng.weibull(1.8,n)",
    "normal_observation_lss": "rng.normal(.3+.4*x,.4,n)",
    "bccg_lss": "DistributionPrediction(BCCGLaw(), {'mu': np.exp(.2+.3*x), 'sigma': np.full(n,.25), 'nu': np.ones(n)}).sample(1, random_state=rng)[0]",
    "beinf_lss": "np.where(rng.uniform(size=n)<.2, 0., np.where(rng.uniform(size=n)<.25, 1., rng.beta(3,5,n)))",
    "zaga_lss": "np.where(rng.uniform(size=n)<.3, 0., rng.gamma(4, np.exp(.3+.4*x)/4))",
    "betabinomial_lss": "rng.binomial(10, rng.beta(2,3,n))",
    "student_lss": "rng.standard_t(5, n)*np.exp(-.5+.2*x)+.3+.5*x",
    "beta_lss": "rng.beta(3,6,n)",
    "nb_lss": "rng.negative_binomial(3,1/(1+np.exp(.3+.5*x)/3))",
    "lognormal_lss": "rng.lognormal(-.2+.5*x, .35, n)",
    "weibull_lss": "np.exp(.2+.3*x)*rng.weibull(1.8,n)",
}

# Observation recipes are authored alongside response recipes, never inferred
# from predictor count or a general promise of context support.
FAMILY_CONTEXT = {"betabinomial_lss": "{'trials': np.full(n,10)}"}
FAMILY_FORMULAS = {"bccg_lss": "{'mu': 'y ~ s(x,k=5)', 'sigma': '~1', 'nu': '~1'}"}
FAMILY_IMPORTS = {"normal_observation_lss":"\nfrom nampy.distributions import ObservationTarget","bccg_lss": "\nfrom nampy.distributions import BCCGLaw, DistributionPrediction"}
FAMILY_FIT_OPTIONS = {"bccg_lss": "smoothing_optimizer='outer_newton'"}

for _name, _law in (("bct_lss", "BCTLaw"), ("bcpe_lss", "BCPELaw")):
    FAMILY_FORMULAS[_name] = "{'mu': 'y~x', 'sigma': '~1', 'nu': '~1', 'tau': '~1'}"
    FAMILY_IMPORTS[_name] = f"\nfrom nampy.distributions import {_law}, DistributionPrediction"
    FAMILY_FIT_OPTIONS[_name] = "optimize_smoothing=False"
for _name, _parameters in (
    ("lognormal_observation_lss", ("loc", "scale")),
    ("weibull_observation_lss", ("scale", "shape")),
):
    FAMILY_FORMULAS[_name] = repr(dict(zip(_parameters, ("~x", "~1"), strict=True)))
    FAMILY_IMPORTS[_name] = "\nfrom nampy.distributions import ObservationTarget"
    FAMILY_FIT_OPTIONS[_name] = "optimize_smoothing=False"

DESIGN_EXAMPLE = (
    BASE
    + """
from nampy.gam import compile_design

design = compile_design(data, formula='y ~ s(x,k=6)+offset(z)', smoothing_params=[.2])
evaluation = design.evaluate(data.assign(z=data.z+.5))
assert design.response.shape == (n,)
assert evaluation.matrix.shape == design.matrix.shape
result = {'matrix': design.matrix, 'penalty': design.penalty,
          'training_offset': design.training_offset, 'evaluation': evaluation,
          'terms': design.terms}
"""
)
FUNCTIONS.update(
    {
        "compile_design": (
            "Compile a statistical design without fitting",
            DESIGN_EXAMPLE,
        ),
        "CompiledDesign": ("Reusable compiled design", DESIGN_EXAMPLE),
        "DesignEvaluation": ("Evaluated design and offsets", DESIGN_EXAMPLE),
        "DesignTerm": ("Compiled term ownership", DESIGN_EXAMPLE),
    }
)


def generate(root):
    """Generate public pages and their downloadable scripts."""
    import inspect

    from feature_reference import heading, statistical_estimator_exports, tree, write
    from public_reference import table

    from nampy import models
    from nampy.gam.design import __all__ as design_symbols
    from nampy.gam.inference.functionals import __all__ as inference_symbols
    from nampy.gam.losses import __all__ as loss_symbols
    from nampy.gam.structured import __all__ as structured_symbols
    survival_symbols = ["piecewise_exponential_data", "survival_from_hazard"]
    extension_symbols = [*loss_symbols, *inference_symbols, *structured_symbols, *survival_symbols]
    from nampy.gam.families.registry import GENERAL_FAMILIES

    root = Path(root)
    if set(FUNCTIONS) != set(extension_symbols) | set(design_symbols):
        raise ValueError(
            "Statistical extension documentation/example inventory differs from runtime exports."
        )
    families = {
        name: spec
        for name, spec in GENERAL_FAMILIES.items()
        if spec.reference_group == "distributional_extensions"
    }
    if set(families) != set(FAMILY_RESPONSES):
        raise ValueError("Every extension family needs an authored example response.")
    for name, module in statistical_estimator_exports().items():
        if module not in {".expectile", ".quantile"}:
            continue
        signature = inspect.signature(getattr(models, name))
        defaults = {
            key: repr(parameter.default)
            for key, parameter in signature.parameters.items()
        }
        page = f"statistical_reference/estimators/{name}"
        code = ASYMMETRIC.format(name=name, extra="")
        text = (
            heading(name)
            + "Types, defaults and output shapes are specified below. Smoothing is fixed; use validation loss to choose its strength.\n\n"
        )
        from feature_reference import signature as render_signature
        from loss_estimator_reference import method_contracts, shared_solver_page

        text += render_signature(getattr(models, name), name)
        shared_solver_page(root)
        parameters = [
            (
                "formula",
                "str | None",
                "None",
                "Response formula with ordinary smooths, tensors and parametric terms. None constructs smooths from numeric input columns.",
            ),
            (
                "level",
                "float",
                "0.5",
                "Level strictly inside (0,1); 0.5 targets the conditional mean."
                if name == "ExpectileGAM"
                else "Level strictly inside (0,1); 0.5 targets the conditional median.",
            ),
            (
                "smoothing_params",
                "float | sequence[float] | mapping | None",
                "None",
                "Fixed nonnegative strengths, one per compiled penalty. None uses ones; this does not run REML.",
            ),
            (
                "k",
                "int",
                "10",
                "Default basis dimension for implicit smooths; explicit formula k values take precedence.",
            ),
            (
                "basis",
                "str",
                "'tp'",
                "Default ordinary smooth basis code. Transformed shape-constrained bases are rejected.",
            ),
            (
                "fit_intercept",
                "bool",
                "True",
                "Include an unpenalized predictor intercept.",
            ),
            (
                "max_iter",
                "int",
                defaults["max_iter"],
                "Positive upper limit on coefficient iterations. Nonconvergence is reported by warning and ``fit_result_.converged``.",
            ),
            (
                "tol",
                "float",
                "1e-9",
                "Positive scaled objective-gradient tolerance for LAWS."
                if name == "ExpectileGAM"
                else "Positive primal/dual residual tolerance for exact pinball, or gradient tolerance for smooth ELF; see the shared solver conventions.",
            ),
        ]
        if "bandwidth" in defaults:
            parameters.append(
                (
                    "bandwidth",
                    "float",
                    "0.0",
                    "Nonnegative response-scale ELF bandwidth. Zero fits exact pinball; positive values smooth the loss, with fixed learning rate one.",
                )
            )
        parameters = [
            (key, kind, defaults[key], description)
            for key, kind, _, description in parameters
        ]
        text += heading("Constructor parameter details", "-") + table(
            ["Parameter", "Type", "Default", "Meaning"],
            [(f"``{key}``", kind, default, meaning) for key, kind, default, meaning in parameters]
        )
        text += "The coefficient vector includes the intercept. The objective is a **sum** of observation losses plus half the coefficient penalty; changing weights changes the effective regularization. Independent levels can cross. Neither Gaussian standard errors nor calibrated qgam intervals are supplied.\n\n"
        text += "For smoothing selection, use ``sklearn.model_selection.GridSearchCV`` with ``param_grid={'smoothing_params': [[.01], [.1], [1.]]}`` and these estimators' negative-loss ``score``. The basis is fitted inside each fold.\n\n"
        text += method_contracts(getattr(models, name))
        code += "score = model.score(data, data.y)\ndesign = model.lpmatrix(data)\ncomponents.validate_additive_reconstruction()\nassert design.shape[1] == model.coef_.size\nprint(score)\n"
        text += example_rst(root, page, code)
        if name == "QuantileGAM":
            smooth_code = code.replace(
                "smoothing_params=[.1])", "smoothing_params=[.1], bandwidth=.1)"
            )
            text += (
                heading("Smooth ELF alternative", "-")
                + "A positive bandwidth changes the training loss; validation score still measures pinball loss.\n\n"
                + example_rst(root, page + "_smooth", smooth_code)
            )
        write(root, page + ".rst", text)
    for name, (title, code) in FUNCTIONS.items():
        page = "statistical_reference/extensions/" + name
        directive = (
            "autoclass"
            if name
            in {
                "DistributedLagBasis",
                "LossFit",
                "FunctionalInterval",
                "CompiledDesign",
                "DesignEvaluation",
                "DesignTerm",
            }
            else "autofunction"
        )
        module = ("nampy.gam" if name in design_symbols else
                  "nampy.gam.losses" if name in loss_symbols else
                  "nampy.gam.structured" if name in structured_symbols else
                  "nampy.survival" if name in survival_symbols else
                  "nampy.gam.inference.functionals")
        text = (
            heading(title)
            + f"Types, defaults and output shapes\n---------------------------------\n\n.. {directive}:: {module}.{name}\n\n"
        )
        if name in {"linear_functional_interval", "derivative_interval"}:
            text += "Simultaneous intervals use joint coefficient deviations and the type-8 empirical quantile, matching gratia's simulation kernel. Coverage applies to the finite evaluation grid and the stated approximate coefficient distribution; it excludes new-response variation.\n\n"
        text += example_rst(root, page, code)
        write(root, page + ".rst", text)
    write(
        root,
        "statistical_reference/extensions/index.rst",
        tree(
            "Statistical extensions",
            [
                name
                for name in FUNCTIONS
                if name
                not in {
                    "compile_design",
                    "CompiledDesign",
                    "DesignEvaluation",
                    "DesignTerm",
                }
            ],
            "Asymmetric objectives, joint uncertainty and EDF, then structured survival and exposure-history models. See :doc:`/development/neighbor_parity` for upstream checks and the exact supported scope.",
            related_links=False,
        ),
    )
    for name, spec in families.items():
        cls = spec.factory.__name__
        names = tuple(spec.factory.parameter_names)
        natural_names = tuple(getattr(spec.factory, "distribution_parameter_names", ()) or names)
        formulas = "{" + ",".join(f"{parameter!r}:" + repr("y ~ s(x,k=5)" if k == 0 else "~x") for k, parameter in enumerate(names)) + "}"
        formulas = FAMILY_FORMULAS.get(name, formulas)
        response = FAMILY_RESPONSES[name]
        context = ", context=" + FAMILY_CONTEXT[name] if name in FAMILY_CONTEXT else ""
        fit_options = FAMILY_FIT_OPTIONS.get(name, "optimize_smoothing=False, smoothing_params=[.2]")
        page = "statistical_reference/families/" + name
        code = f"""import numpy as np
import pandas as pd
from nampy import GAMLSS{FAMILY_IMPORTS.get(name, '')}

rng = np.random.default_rng(17)
n = 120
x = np.linspace(-1,1,n)
data = pd.DataFrame({{'x': x, 'y': {response}}})
model = GAMLSS(family={name!r}, formula={formulas},
               {fit_options}).fit(data{context})
parameters = model.predict_parameters(data)
point = model.predict_point(data{context})
assert parameters.shape == (n,{len(natural_names)})
result = {{'parameters': parameters, 'point': point, 'components': model.predict_components(data)}}
"""
        if name in {"bct_lss", "bcpe_lss"}:
            # Balanced quantiles make the four-parameter example reproducible
            # without relying on a lucky small-sample tail realization.
            code = code.replace("default_rng(17)", "default_rng(41)").replace("n = 120", "n = 160")
        if name == "normal_observation_lss":
            code = code.replace("model = GAMLSS", "target = ObservationTarget(np.where(data.y<0,-np.inf,data.y),data.y)\nmodel = GAMLSS").replace(".fit(data)", ".fit(data,target)")
        elif name in {"lognormal_observation_lss", "weibull_observation_lss"}:
            # Select a fixed truncated response population, then record censoring
            # explicitly in the shared target contract supplied to fit.
            code = code.replace(
                "model = GAMLSS",
                "data = data.loc[(data.y>0.2) & (data.y<3)].copy()\n"
                "n = len(data)\n"
                "target = ObservationTarget(\n"
                "    np.where(data.y<0.7, -np.inf, np.minimum(data.y, 1.8)),\n"
                "    np.where(data.y>1.8, np.inf, np.maximum(data.y, 0.7)))\n"
                "model = GAMLSS",
            ).replace(f"family={name!r}", f"family={{'name': {name!r}, 'lower': 0.2, 'upper': 3.0}}")
            code = code.replace(".fit(data)", ".fit(data[['x']], target)")
        text = heading(name)
        from family_contracts import mathematical_contract

        text += mathematical_contract(name, "statistical")
        text += heading("Types, defaults and output shapes", "-")
        text += f".. autoclass:: {spec.factory.__module__}.{cls}\n\n"
        text += (
            "Pass ``{'name': 'student_lss', 'df': 5.0}`` to configure fixed degrees of freedom; df must be finite and >2.\n\n"
            if name == "student_lss"
            else f"Use family={{'name': {name!r}, 'lower': L, 'upper': U}} for fixed truncation and supply ObservationTarget to fit. Defaults leave both tails untruncated.\n\n" if name.endswith("_observation_lss") else f"Pass ``GAMLSS(family={name!r})``; this family has no constructor options.\n\n"
        )
        point_summary = "conditional median" if name in {"lognormal_observation_lss", "weibull_observation_lss"} else "conditional mean"
        text += f"``predict_parameters(X)`` returns natural parameters as a float ``(n,{len(natural_names)})`` array, ``predict_link(X)`` returns link predictors ``(n,{len(names)})``, ``predict_point(X)`` returns the {point_summary} ``(n,)``, and ``predict_components(X)`` returns an AdditivePrediction whose link and each term are ``(n,{len(names)})``. ``X`` is a DataFrame or numeric ``(n,d)`` array, consistent with formula mode.\n\n"
        derivative_order = {2: 'two', 3: 'three', 4: 'four'}[spec.factory.max_derivative_order]
        text += f"Likelihood derivatives through order {derivative_order} are analytic, including mixed derivatives. Fixed smoothing is tested; automatic smoothing requires an enabled runtime capability and route-specific evidence. Residual availability follows the observation contract; ordinary response/Pearson residuals are unavailable for censored targets. Deviance residuals and null deviance are not defined by this implementation; use explicit response/Pearson diagnostics or shared distribution quantile residuals rather than the default deviance-residual plotting workflow.\n\n"
        text += example_rst(root, page, code)
        write(root, page + ".rst", text)
