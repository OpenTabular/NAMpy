"""Authored contracts for statistical links and the helper APIs behind a GAM."""

import inspect

LINKS = {
    "identity": (
        r"g(\mu)=\mu,\qquad g^{-1}(\eta)=\eta",
        r"\mu,\eta\in\mathbb R",
        r"d\mu/d\eta=1",
        "gaussian",
        "An additive predictor difference is an additive mean difference. This leaves the response family's domain restrictions to the family; identity does not by itself keep Poisson or Gamma means positive.",
        "No clipping or positivity transform is applied.",
    ),
    "log": (
        r"g(\mu)=\log\mu,\qquad g^{-1}(\eta)=e^\eta",
        r"\mu>0,\quad\eta\in\mathbb R",
        r"d\mu/d\eta=e^\eta",
        "poisson",
        "Exponentiating a term difference gives a conditional mean ratio. A log link changes how the mean depends on inputs; it does not assume that log(Y) is Gaussian.",
        "The inverse has a machine-epsilon lower floor. Large positive predictors can overflow exp; forward log requires a positive input and does not clip invalid means.",
    ),
    "inverse": (
        r"g(\mu)=1/\mu,\qquad g^{-1}(\eta)=1/\eta",
        r"\mu\ne0,\quad\eta\ne0",
        r"d\mu/d\eta=-1/\eta^2",
        "gamma",
        "This is a decreasing mean link. Positive-response families require the positive branch; a Gaussian inverse-link model can use a nonzero negative branch where its family permits it. Reciprocal effects are not mean ratios.",
        "No singularity removal is performed. Zero inputs produce nonfinite reciprocal values; satisfy family.valid_mu and family.valid_eta before evaluating.",
    ),
    "sqrt": (
        r"g(\mu)=\sqrt\mu,\qquad g^{-1}(\eta)=\eta^2",
        r"\mu\geq0,\quad\eta\geq0",
        r"d\mu/d\eta=2\eta",
        "poisson",
        "Effects are additive in the square root of the mean. The principal inverse requires a nonnegative predictor; the algebraic square alone does not validate that branch. Fitting uses the positive interior where link derivatives are finite.",
        "inverse_link squares its input without enforcing the principal branch. Higher forward derivatives are singular at a zero mean. Use the family's domain checks.",
    ),
    "inverse_squared": (
        r"g(\mu)=\mu^{-2},\qquad g^{-1}(\eta)=\eta^{-1/2}",
        r"\mu>0,\quad\eta>0",
        r"d\mu/d\eta=-\tfrac12\eta^{-3/2}",
        "inverse_gaussian",
        "The canonical inverse-Gaussian mean link uses the positive inverse-square-root branch. This is a decreasing link: a larger additive predictor implies a smaller mean.",
        "Use link='1/mu^2'. Nonpositive predictors are outside the intended inverse domain; the helper does not replace them with positive values.",
    ),
    "logit": (
        r"g(p)=\log\frac{p}{1-p},\qquad g^{-1}(\eta)=\frac1{1+e^{-\eta}}",
        r"0<p<1,\quad\eta\in\mathbb R",
        r"dp/d\eta=p(1-p)",
        "binomial",
        "A term difference is a difference in log odds, so its exponential is an odds ratio conditional on other inputs. An odds ratio is not a probability difference or a risk ratio.",
        "The inverse and its derivative use the stats-style strict +/-30 tail thresholds and machine-epsilon safeguards. Forward evaluation requires an interior probability.",
    ),
    "probit": (
        r"g(p)=\Phi^{-1}(p),\qquad g^{-1}(\eta)=\Phi(\eta)",
        r"0<p<1,\quad\eta\in\mathbb R",
        r"dp/d\eta=\varphi(\eta)",
        "binomial",
        "Phi is the standard normal CDF and varphi its density. This link can be interpreted through a Gaussian latent threshold. Coefficients are on that latent scale and are not log odds ratios.",
        "The inverse probability is clipped to the machine-epsilon interior; the normal-density derivative also has a positive floor. Forward quantiles at exact probability endpoints are infinite.",
    ),
    "cloglog": (
        r"g(p)=\log\{-\log(1-p)\},\qquad g^{-1}(\eta)=1-e^{-e^\eta}",
        r"0<p<1,\quad\eta\in\mathbb R",
        r"dp/d\eta=\exp\{\eta-e^\eta\}",
        "binomial",
        "The complementary log-log link is asymmetric. Its exponential predictor is a cumulative-hazard-like quantity when a binary event model has that interpretation. It is not the logit link with a different numerical optimizer.",
        "The inverse uses expm1 and clips probabilities to the interior. mu_eta caps its exponential argument at 700 and floors the result at machine epsilon; very large inverse-link arguments can still emit an exponential-overflow warning.",
    ),
    "cauchit": (
        r"g(p)=\tan\{\pi(p-\tfrac12)\},\qquad g^{-1}(\eta)=\tfrac12+\frac{\arctan\eta}{\pi}",
        r"0<p<1,\quad\eta\in\mathbb R",
        r"dp/d\eta=\frac1{\pi(1+\eta^2)}",
        "binomial",
        "A Cauchy latent threshold gives probabilities with heavier predictor tails than a probit link. This changes the response to extreme predictor values; it does not make the binary response a Student-t distribution.",
        "Forward evaluation uses stable cotangent/complement forms. The inverse limits predictor magnitude using the machine-epsilon tail quantile, and the inverse derivative has a positive floor.",
    ),
    "power": (
        r"g(\mu)=\mu^a,\qquad g^{-1}(\eta)=\eta^{1/a}\quad(a>0,\ a\ne1)",
        r"\mu>0,\quad\eta>0,\quad0<a<\infty",
        r"d\mu/d\eta=\tfrac1a\eta^{1/a-1}",
        "quasi",
        "The exponent changes the link, independently of the quasi variance function. The public quasi factory dispatches finite power<=0 to a log link and power=1 to identity; only positive nonunit exponents construct PowerLink. A negative exponent does not select an inverse-power link through this option.",
        "Set family={'name':'quasi','variance':'mu^3','power':0.5}; link may be omitted, 'identity', or 'power' when power is supplied. Combining it with another named link raises ValueError. A 'power' link without a numeric power is invalid. Inverse values and inverse derivatives have machine-epsilon lower floors.",
    ),
}


def field_list(entries):
    """Definition lists keep parameters/outputs detailed without output tables."""
    text = ""
    for name, kind, default, meaning in entries:
        text += f"``{name}`` — Type: {kind}\n   "
        if default is not None:
            text += f"Default: ``{default}``. "
        text += meaning + "\n\n"
    return text


def contract(obj, parameters, outputs, *, spelling=None, level="-"):
    from feature_reference import heading

    name = spelling or obj.__name__
    return (
        heading("Signature", level)
        + f".. code-block:: python\n\n   {name}{inspect.signature(obj)}\n\n"
        + heading("Parameters", level)
        + (
            field_list(parameters)
            if parameters
            else "The constructor takes no arguments.\n\n"
        )
        + heading("Outputs", level)
        + field_list(outputs)
    )


def link_code(slug):
    _, _, _, family, _, _ = LINKS[slug]
    key = "1/mu^2" if slug == "inverse_squared" else slug
    options = (
        {"name": family, "link": key}
        if slug != "power"
        else {"name": "quasi", "variance": "mu^3", "power": 0.5}
    )
    values = (
        "[0.1, 0.5, 0.9]"
        if slug in {"logit", "probit", "cloglog", "cauchit"}
        else "[0.5, 1.0, 2.0]"
    )
    return f"""import numpy as np
from nampy.gam.families.registry import make_gam_family

family = make_gam_family({options!r})
mean = np.array({values})
predictor = family.link(mean)
recovered_mean = family.inverse_link(predictor)
inverse_derivative = family.mu_eta(predictor)
second_forward_derivative = family.d2link(mean)
result = {{"predictor": predictor, "recovered_mean": recovered_mean,
          "inverse_derivative": inverse_derivative,
          "second_forward_derivative": second_forward_derivative}}
"""


def generate_links(root):
    from feature_reference import heading, tree, write
    from reference_examples import example_rst

    from nampy.gam.families._function_maps import LINK_REGISTRY
    from nampy.gam.families.registry import make_gam_family

    expected = {
        "1/mu^2" if name == "inverse_squared" else name
        for name in LINKS
        if name != "power"
    }
    if set(LINK_REGISTRY) != expected:
        raise ValueError("Every named statistical link requires an authored contract")
    for slug, (
        formula,
        domain,
        derivative,
        family_name,
        interpretation,
        numerical,
    ) in LINKS.items():
        key = "1/mu^2" if slug == "inverse_squared" else slug
        text = heading(
            "Inverse-squared link"
            if slug == "inverse_squared"
            else slug.capitalize() + " link"
        )
        text += f".. math::\n\n   {formula}\n\n* **Domains:** :math:`{domain}`. The response family can impose stricter restrictions.\n* **Inverse derivative:** :math:`{derivative}`.\n\n"
        text += interpretation + "\n\n" + numerical + "\n\n"
        options = (
            {"name": family_name, "link": key}
            if slug != "power"
            else {"name": "quasi", "variance": "mu^3", "power": 0.5}
        )
        text += (
            heading("Configure the family", "-")
            + f".. code-block:: python\n\n   from nampy.gam.families.registry import make_gam_family\n\n   family = make_gam_family({options!r})\n   # Pass this instance, or its options mapping, to GAM(family=...).\n\n"
        )
        text += "The family factory has signature ``make_gam_family(family)``. Its required argument accepts a family name, options mapping, family instance (cloned), or ``None`` (Gaussian identity); it returns one independent family object. The mapping's ``name`` and ``link`` are strings. Unsupported family/link pairs raise ``ValueError``. The numerical link classes are implementation objects obtained through a family, rather than a second public factory.\n\n"
        if slug == "power":
            text += "``power`` is a finite real scalar, default ``None``; booleans and nonfinite values are rejected. ``variance`` defaults to ``'constant'`` and independently selects one of ``constant``, ``mu``, ``mu(1-mu)``, ``mu^2``, ``mu^3``. The options mapping accepts numeric power for general ``quasi``; named quasi-Poisson/binomial factory mappings accept only ``link`` and keep their own variance functions. See :doc:`../families/quasi`.\n\n"
        else:
            accepted = []
            for name, extra in [
                (n, {})
                for n in [
                    "gaussian",
                    "poisson",
                    "binomial",
                    "gamma",
                    "inverse_gaussian",
                    "quasi",
                    "quasipoisson",
                    "quasibinomial",
                    "nb",
                    "betar",
                    "scat",
                    "zip",
                    "cnorm",
                    "clog",
                    "cpois",
                    "bcg",
                    "tw",
                ]
            ] + [("negbin", {"theta": 2.0}), ("ocat", {"R": 3})]:
                try:
                    family = make_gam_family({"name": name, "link": key, **extra})
                except (ValueError, TypeError):
                    continue
                if isinstance(family.link, LINK_REGISTRY[key]):
                    accepted.append(f":doc:`{name} <../families/{name}>`")
            text += (
                "Single-predictor family factories using this named link: "
                + ", ".join(accepted)
                + ". This describes link selection, not every family/design/optimizer combination. Specialized distributional links retain their family-specific parameterization.\n\n"
            )
        text += heading("Evaluation signatures and inputs", "-")
        # The callable link object is installed on the family instance.
        demo = make_gam_family(options)
        text += ".. code-block:: python\n\n"
        text += f"   family.link{inspect.signature(demo.link.__call__)}\n"
        for name in ["inverse_link", "mu_eta", "d2link", "d3link", "d4link"]:
            text += f"   family.{name}{inspect.signature(getattr(demo, name))}\n"
        text += "\n" + field_list(
            [
                (
                    "mu",
                    "array-like of floats, shape ``S`` (scalar ``()`` permitted)",
                    "required",
                    "Response means/probabilities in the displayed link domain. Convert input to a numeric array; the forward link does not generally repair invalid domains.",
                ),
                (
                    "eta",
                    "array-like of floats, shape ``S`` (scalar ``()`` permitted)",
                    "required",
                    "Additive predictors in the inverse-link domain. ``inverse_link`` and ``mu_eta`` consume eta, not response means.",
                ),
            ]
        )
        text += heading("Outputs", "-") + field_list(
            [
                (
                    "family.link(mu)",
                    "floating NumPy array/scalar-shaped numerical result, shape ``S``",
                    None,
                    "Forward predictor values. An existing float64 input can be returned without copying for identity.",
                ),
                (
                    "family.inverse_link(eta)",
                    "floating NumPy array/scalar-shaped numerical result, shape ``S``",
                    None,
                    "Response mean/probability after the numerical safeguards described above; no response noise is added.",
                ),
                (
                    "family.mu_eta(eta)",
                    "floating numerical result, shape ``S``",
                    None,
                    "Derivative of the inverse link, d mu / d eta. It can be negative for decreasing links.",
                ),
                (
                    "family.d2link(mu), d3link(mu), d4link(mu)",
                    "floating numerical results, shape ``S``",
                    None,
                    "Second, third and fourth derivatives of the forward link with respect to mu, used by fitting derivatives. These are not derivatives of predictions with respect to a source feature.",
                ),
            ]
        )
        text += "Ordinary Python lists are accepted through NumPy conversion. Scalar inputs retain shape ``()``; concrete scalar container types can differ between link operations. These elementwise methods preserve broadcasting shape and do not check model convergence.\n\n"
        text += "Link definitions follow `R's GLM link conventions <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/make.link.html>`_; numerical behavior and accepted factory combinations above are taken from NAMpy.\n\n"
        text += example_rst(
            root, "statistical_reference/links/" + slug, link_code(slug)
        )
        write(root, "statistical_reference/links/" + slug + ".rst", text)
    text = tree(
        "Link functions",
        list(LINKS),
        "A response law and a link are separate choices. The link connects a conditional parameter to an additive predictor; its inverse maps back to that parameter's domain. Each page gives the formula, domains, numerical safeguards, factory configuration, evaluation signatures and a runnable example.",
    )
    text += (
        "\n"
        + heading("Which derivative?", "-")
        + "``mu_eta(eta)`` differentiates the inverse link. ``d2link(mu)`` through ``d4link(mu)`` differentiate the forward link. Numerical family fitting uses these derivatives with the basis design; they are separate from ``GAM.derivative``, which differentiates a fitted term with respect to its covariate.\n\n"
    )
    text += "Changing the link changes effect interpretation, not the response support or variance function. In particular, quasi ``power`` configures a link exponent, while ``variance`` selects the mean-variance relationship. Families with several predictors also have specialized links described on their :doc:`family pages <../families/index>`.\n"
    text += (
        "\n"
        + heading("Family factory", "-")
        + contract(
            make_gam_family,
            [
                (
                    "family",
                    "str | mapping[str, object] | BaseFamily | None",
                    "required (None selects Gaussian identity)",
                    "A name uses that family's default link. A mapping selects name and supported options such as link. An existing family is independently cloned. Unsupported names/options or links raise errors according to the chosen factory; custom R closures are not accepted.",
                )
            ],
            [
                (
                    "return value",
                    "BaseFamily subclass instance, scalar object",
                    None,
                    "Unfitted, independent response-family state. It supplies link, inverse_link, mu_eta and forward-link derivatives where applicable; it does not contain fitted regression predictions.",
                )
            ],
            level="~",
        )
    )
    defaults = []
    for name in [
        "gaussian",
        "binomial",
        "poisson",
        "gamma",
        "inverse_gaussian",
        "quasi",
        "quasipoisson",
        "quasibinomial",
    ]:
        family = make_gam_family(name)
        key = next(
            key for key, cls in LINK_REGISTRY.items() if isinstance(family.link, cls)
        )
        slug = "inverse_squared" if key == "1/mu^2" else key
        defaults.append(
            f"* :doc:`{name} <../families/{name}>`: :doc:`{key} <{slug}>`.\n"
        )
    text += heading("Default links", "-") + "".join(defaults) + "\n"
    write(root, "statistical_reference/links/index.rst", text)


def helper_specs():
    from nampy.gam.diagnostics.plots import exclude_too_far
    from nampy.gam.fit.selection import (
        n_free_smoothing_params,
        optimizer_endpoint_diagnostics,
        resolve_smoothing_method,
        supports_smoothing_method,
    )
    from nampy.gam.inference.chi_square_mixtures import (
        DaviesAlgorithm,
        liu2,
        psum_chisq,
    )

    return {
        fn.__name__: fn
        for fn in [
            resolve_smoothing_method,
            supports_smoothing_method,
            n_free_smoothing_params,
            optimizer_endpoint_diagnostics,
            exclude_too_far,
            psum_chisq,
            liu2,
            DaviesAlgorithm,
        ]
    }


def helper_example(name):
    from reference_examples import BASE, MODEL

    if name in {
        "optimizer_endpoint_diagnostics",
        "supports_smoothing_method",
        "resolve_smoothing_method",
        "n_free_smoothing_params",
    }:
        code = BASE + MODEL + f"from nampy.gam.fit.selection import {name}\n"
        if name == "optimizer_endpoint_diagnostics":
            code += "diagnostic = optimizer_endpoint_diagnostics(model)\nresult = {'projected_gradient': np.asarray(diagnostic['projected_gradient']), 'stationary': diagnostic['stationary_by_projected_gradient'], 'flat_ridge_suspected': diagnostic['flat_ridge_suspected'], 'free_parameters': diagnostic['n_free_smoothing_params']}\n"
        elif name == "supports_smoothing_method":
            code += "result = {method: supports_smoothing_method(model, method) for method in ['fixed', 'gcv', 'ubre', 'reml']}\n"
        elif name == "resolve_smoothing_method":
            code += "result = {str(request): resolve_smoothing_method(model, request) for request in [None, 'auto', 'gcv.cp', 'REML']}\n"
        else:
            code += "partly_fixed = GAM(formula=\"y ~ s(x, bs='cr', k=6, sp=0.2) + s(z, bs='cr', k=6)\", optimize_smoothing=False).fit(data=data)\nresult = {'all_free': n_free_smoothing_params(model), 'partly_fixed': n_free_smoothing_params(partly_fixed), 'fixed_mask': partly_fixed.smoothing_fixed_mask_}\n"
        return code
    if name == "exclude_too_far":
        return """import numpy as np
from nampy.gam.diagnostics.plots import exclude_too_far

gx, gz = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
observed_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
observed_z = observed_x.copy()
mask = exclude_too_far(gx.ravel(), gz.ravel(), observed_x, observed_z, dist=0.2)
surface_mask = mask.reshape(gx.shape)
result = {"excluded": int(mask.sum()), "retained": int((~mask).sum()), "surface_mask": surface_mask}
"""
    if name == "DaviesAlgorithm":
        return """import numpy as np
from nampy.gam.inference.chi_square_mixtures import DaviesAlgorithm

probability, trace, status = DaviesAlgorithm().davies(
    lb=np.array([0.5, 1.5]), nc=np.zeros(2), n=np.array([1, 2]),
    r=2, sigma=0.0, c_val=4.0, lim=100000, acc=2e-5,
)
result = {"lower_tail_probability": probability, "status": status, "trace": trace}
"""
    return f"""import numpy as np
from nampy.gam.inference.chi_square_mixtures import {name}

thresholds = np.array([1.0, 4.0, 8.0])
weights = np.array([0.5, 1.5])
degrees = np.array([1, 2])
upper = {name}(thresholds, weights, df=degrees)
lower = {name}(thresholds, weights, df=degrees, lower_tail=True)
result = {{"thresholds": thresholds, "upper_tail": upper, "lower_tail": lower}}
"""


def helper_contracts(name):
    model = (
        "model",
        "fitted core ``nampy.gam.GAM``",
        "required",
        "Use a fitted core model, or ``estimator.gam_`` from a wrapper. Required compiled/fitted state is read from this object; passing a wrapper directly is not supported.",
    )
    if name == "optimizer_endpoint_diagnostics":
        parameters = [
            model,
            (
                "conv_tol",
                "float > 0",
                "1e-6",
                "Projected-gradient stationarity tolerance, scaled by 1 + abs(criterion score). Supply a positive finite value; this helper does not implement a general input validator.",
            ),
            (
                "fd_step",
                "float > 0",
                "1e-3",
                "Candidate central-difference step along a common log-smoothing shift. Bound distances can reduce it; no feasible positive step gives None finite-difference diagnostics.",
            ),
        ]
        output = [
            (
                "return value",
                "dict[str, object] | None",
                None,
                "None for a fixed/no-selected criterion, no penalties, or no free smoothing coordinates. An unfitted core model raises an error. Otherwise the fields below describe the current endpoint, not a newly fitted model.",
            ),
            (
                "criterion_name; criterion_backend",
                "str; str | None",
                None,
                "Resolved fitted criterion and its available ML/REML backend. A None backend for a non-likelihood criterion does not itself indicate fit failure.",
            ),
            (
                "optimizer_success; optimizer_message",
                "bool | None; str | None",
                None,
                "Recorded optimizer result, or None when no result record exists. Recorded success and stationarity diagnostics answer different questions.",
            ),
            (
                "n_free_smoothing_params",
                "int s",
                None,
                "Number of free log smoothing coordinates.",
            ),
            (
                "log_smoothing_params; gradient; projected_gradient",
                "list[float], each length s",
                None,
                "Current free log penalties, raw criterion gradient, and bound-projected gradient in the same coordinate order.",
            ),
            (
                "bounds",
                "list[list[float]], shape (s,2)",
                None,
                "min_sp-derived lower log bounds and unbounded upper ends. Infinite entries are possible. This is not a complete echo of every numerical optimizer's sp_log_bounds.",
            ),
            (
                "gradient_inf_norm; projected_gradient_inf_norm",
                "float",
                None,
                "Largest absolute raw/projected gradient coordinate.",
            ),
            (
                "stationary_by_raw_gradient; stationary_by_projected_gradient; boundary_limited",
                "bool",
                None,
                "Tolerance decisions. boundary_limited indicates a projected stationary endpoint whose raw gradient still exceeds tolerance at a reported bound.",
            ),
            (
                "at_lower_bound; at_upper_bound",
                "list[bool], each length s",
                None,
                "Whether coordinates meet the reported bound tests.",
            ),
            (
                "hessian; hessian_eigenvalues; min_abs_hessian_eigenvalue",
                "list[list[float]] (s,s) | None; list[float] (s,) | None; float | None",
                None,
                "Criterion curvature, its symmetric eigenvalues and the smallest eigenvalue magnitude. None means unavailable curvature, not zero curvature.",
            ),
            (
                "shared_shift_directional_derivative; shared_shift_curvature",
                "float; float | None",
                None,
                "Gradient and analytic curvature along the normalized all-ones direction in free log-penalty coordinates.",
            ),
            (
                "shared_shift_fd_step; shared_shift_fd_slope; shared_shift_fd_curvature",
                "float | None",
                None,
                "Feasible central-difference step and slope/curvature checks. These can be absent at a bound.",
            ),
            (
                "flat_ridge_suspected",
                "bool",
                None,
                "Heuristic combining small projected gradient and weak Hessian/shared-direction curvature. It is not a proof of model nonidentifiability.",
            ),
            (
                "factor_smooth_shared_ridge_stabilized; factor_smooth_shared_ridge_shift",
                "bool; float | None",
                None,
                "Factor-smooth stabilization flag and optional shift, where supplied by the route.",
            ),
            (
                "joint_gaussian_reml_outer; joint_negbin_reml_outer; joint_negbin_efs_outer; joint_negbin_postprocessed; joint_negbin_flat_ridge_stabilized",
                "bool",
                None,
                "Route-specific flags for joint scale/negative-binomial nuisance fitting and postprocessing.",
            ),
            (
                "joint_log_theta; joint_negbin_initial_log_theta; family_theta",
                "float | None",
                None,
                "Optional current/initial log nuisance dispersion and natural scalar family theta.",
            ),
            (
                "joint_negbin_optimizer_message; joint_negbin_optimizer_fun; joint_negbin_optimizer_nfev; joint_negbin_optimizer_njev",
                "str | None; float | None; int | None; int | None",
                None,
                "Optional joint negative-binomial optimizer message, objective and evaluation counts.",
            ),
        ]
        return parameters, output
    if name == "supports_smoothing_method":
        return [
            model,
            (
                "method",
                "str",
                "required",
                "Canonical lower-case method: fixed, gcv, gacv, ubre, aic, ubreaic, ml, reml, p-ml, p-reml, laml, ncv or qncv. This helper does not lowercase or resolve aliases; unknown values raise ValueError.",
            ),
        ], [
            (
                "return value",
                "bool, shape ()",
                None,
                "Reports this model/family criterion capability, including scale, transformed-coefficient/observation and derivative guards. It does not certify every optimizer, formula composition or convergence condition; fitting performs further validation.",
            )
        ]
    if name == "resolve_smoothing_method":
        return [
            model,
            (
                "method",
                "str | None",
                "required (None is allowed)",
                "None or 'auto' selects a supported default. Strings are lowercased; GCV.Cp/GACV.Cp and likelihood spellings follow criterion normalization. Unsupported names raise ValueError.",
            ),
        ], [
            (
                "return value",
                "str, scalar",
                None,
                "Resolved method name. auto prefers available REML, then known-scale ubreaic, then GCV, then fixed. Extended-family predictive criterion spellings can normalize to reml; the result is a criterion name, not an optimizer or a completed fit.",
            )
        ]
    if name == "n_free_smoothing_params":
        return [model], [
            (
                "return value",
                "int >= 0, shape ()",
                None,
                "Count of entries not marked fixed in ``smoothing_fixed_mask_``; with no mask, count of model smoothing coordinates. A fixed-SP solve can still have entries marked free: ``optimize_smoothing=False`` does not change this bookkeeping count into zero.",
            )
        ]
    if name == "exclude_too_far":
        return [
            (
                "g1; g2",
                "array-like[float], each flattened to (n_grid,)",
                "required",
                "Paired grid coordinates. Lengths must agree; each grid axis must have finite nonzero range for the normalization to be defined.",
            ),
            (
                "d1; d2",
                "array-like[float], each flattened to (n_data,)",
                "required",
                "Paired observed coordinates; equal lengths and at least one finite observed point. They are normalized using grid bounds, not their own bounds.",
            ),
            (
                "dist",
                "float >= 0",
                "required",
                "Euclidean exclusion radius after scaling the grid axes to [0,1]. It is dimensionless, not a raw distance in either input unit.",
            ),
        ], [
            (
                "return value",
                "numpy.ndarray[bool], shape (n_grid,)",
                None,
                "True marks a grid point whose nearest observed point is strictly farther than dist. False retains it. Reshape explicitly to a mesh shape if needed. Inputs are copied; unequal paired lengths or negative dist raise ValueError. Degenerate/empty/nonfinite grids are not comprehensively validated by the implementation.",
            )
        ]
    if name in {"psum_chisq", "liu2"}:
        params = [
            (
                "q" if name == "psum_chisq" else "x",
                "float | one-dimensional array-like[float], shape () or (n,)",
                "required",
                "Threshold(s) at which the tail probability is evaluated. Use scalar or 1D input; arbitrary multi-axis input is not supported by the Davies wrapper.",
            ),
            (
                "lb",
                "array-like[float], flattened shape (r,)",
                "required",
                "Weights of independent chi-square variables. Davies supports signed weights; at least one nonzero weight is required. For the intended Liu approximation use nonnegative weights with at least one positive entry.",
            ),
            (
                "df",
                "array-like[float] | None, scalar/length 1 or (r,)",
                "None",
                "Degrees of freedom, default ones. A single value is repeated. psum_chisq rounds with NumPy rint and requires positive integer results; liu2 uses supplied real values directly and only checks matching lengths. Supply positive degrees of freedom.",
            ),
        ]
        if name == "psum_chisq":
            params += [
                (
                    "nc",
                    "array-like[float] | None, scalar/length 1 or (r,)",
                    "None",
                    "Noncentralities, default zeros; a single value is repeated. Supply finite nonnegative values. Mismatched vector lengths raise ValueError.",
                ),
                (
                    "sigz",
                    "float",
                    "0.0",
                    "Multiplier of an independent standard normal term. Nonpositive values are treated as zero.",
                ),
                (
                    "lower_tail",
                    "bool",
                    "False",
                    "False requests the upper tail; True requests the CDF on a successful Davies evaluation. Read the fallback qualification below.",
                ),
                (
                    "tol",
                    "float > 0",
                    "2e-5",
                    "Requested numerical integration accuracy; not a statistical confidence level.",
                ),
                (
                    "nlim",
                    "int > 0",
                    "100000",
                    "Maximum integration-term budget passed to Davies.",
                ),
            ]
        else:
            params += [
                (
                    "lower_tail",
                    "bool",
                    "False",
                    "False requests the upper-tail moment approximation; True requests its CDF for positive thresholds. Nonpositive-threshold behavior is noted below.",
                )
            ]
        return params, [
            (
                "return value",
                "float for scalar input; numpy.ndarray[float64] of shape (n,) for vector input",
                None,
                "Tail probability/approximation in input threshold order. The wrapper can return NaN when numerical evaluation fails. No trace attributes are attached; use DaviesAlgorithm.davies for raw integration diagnostics. See the numerical/fallback behavior below before interpreting a returned value.",
            )
        ]
    if name == "DaviesAlgorithm":
        return [], [
            (
                "constructor",
                "DaviesAlgorithm instance",
                None,
                "Creates a numerical integration object; no probabilities are computed until davies is called.",
            )
        ]
    raise KeyError(name)


HELPER_THEORY = {
    "optimizer_endpoint_diagnostics": (
        r"\rho=\log\lambda,\qquad g=\nabla C(\rho),\quad H=\nabla^2C(\rho),\quad\tau=\mathrm{conv\_tol}(1+|C|)",
        "A small coefficient gradient and a small outer smoothing gradient are different conditions. This helper examines the outer criterion at the current endpoint, using recorded derivatives when valid and recomputing them otherwise. It does not choose new smoothing values. At a lower bound a positive gradient can be compatible with a constrained optimum because decreasing that coordinate is infeasible; its projected component is zero. The reported stationarity decisions compare infinity norms with tau.\n\nA stationary point need not be a minimum: negative Hessian eigenvalues indicate nonconvex curvature, while tiny eigenvalues can indicate weakly determined penalty combinations. The common-shift checks move all free log penalties together along a normalized direction, optionally checking derivatives with central differences. ``flat_ridge_suspected`` is a numerical heuristic, not a statistical test. Its bounds are derived from ``min_sp`` and have infinite upper ends; interpret the projection in that convention.\n\nRecorded optimizer success, endpoint stationarity, and scientific model adequacy must be assessed separately. This function evaluates numerical derivatives and can be more expensive than reading a fit-result flag.",
    ),
    "supports_smoothing_method": (
        r"\mathcal C(\text{model},\text{criterion})\in\{\mathrm{True},\mathrm{False}\}",
        "A criterion needs more than a family name: it can require a known dispersion, a particular likelihood backend, derivative orders and compatible coefficient/observation transforms. This helper combines those capability checks. For example, known-scale UBRE and estimated-scale Gaussian REML have different prerequisites.\n\nResolve aliases first with :doc:`resolve_smoothing_method`. The predicate expects lower-case canonical names and does not normalize them. It does not run every design/optimizer compatibility check or fit a model; a True result is a capability indication rather than a convergence guarantee. NCV/QNCV capability also depends on the ordinary-family derivative route, not merely the availability of a neighborhood argument.",
    ),
    "resolve_smoothing_method": (
        r"\text{requested method}\longmapsto\text{normalized fitting criterion}",
        "The requested spelling can represent a family-dependent policy rather than a literal criterion. ``auto`` prefers an available REML backend, then UBRE/AIC with known scale, then GCV, and finally fixed smoothing. GCV.Cp/GACV.Cp resolve through the family/scale criterion convention.\n\nThe resolver lowercases names and applies implemented extended-family and known-scale aliases. For extended families, predictive-score spellings outside their supported likelihood route can normalize to REML. A resolved string does not select an outer optimizer or certify convergence. Use :doc:`supports_smoothing_method` for its capability predicate and the fitting-support matrix for complete route restrictions.",
    ),
    "n_free_smoothing_params": (
        r"s_{\mathrm{free}}=\sum_{j=1}^{s}\mathbf1\{\text{fixed\_mask}_j=\mathrm{False}\}",
        "A term can own several penalties, and several terms can share one smoothing coordinate. The number of free smoothing parameters therefore differs from the number of smooth terms and from coefficient dimension. This helper counts coordinates after the compiled penalty mapping.\n\nSupplied fixed strengths are excluded through the fixed mask. With no mask it returns the total coordinate count. The count is bookkeeping: turning automatic selection off holds the current values for a coefficient solve, but does not necessarily mark every coordinate fixed in this mask. Inspect the optimization flag separately.",
    ),
    "exclude_too_far": (
        r"\mathrm{exclude}_i=\mathbf1\left\{\min_j\left\|\widetilde g_i-\widetilde d_j\right\|_2>\mathrm{dist}\right\}",
        "Prediction over a rectangle can include locations far outside observed support. This helper makes a geometric mask by scaling each grid axis to [0,1], applying those same shifts/scales to observations, and comparing each grid node's nearest observed distance with a radius. Thus changing axis units does not change the mask under a consistent positive linear rescaling.\n\nThe scaling comes from the grid range, not the observed range. Equality at the radius is retained. A True entry means exclude it; it is not a keep mask. This does not compute a convex hull, density threshold or uncertainty interval, and it does not change fitted coefficients. A nondegenerate finite grid and nonempty observations are prerequisites. The implementation forms pairwise distances, so memory grows with grid size times data size.\n\nSee `mgcv exclude.too.far <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/exclude.too.far.html>`_ for the geometric convention. ``plot``/``vis`` expose related masking options; this page documents direct mask construction.",
    ),
    "psum_chisq": (
        r"Q=\sum_{j=1}^{r}\lambda_jX_j+\sigma_zZ,\qquad X_j\sim\chi^2_{h_j}(\delta_j),\quad Z\sim N(0,1)",
        "The components are independent. Weighted chi-square sums arise when a Gaussian quadratic-form statistic is diagonalized: covariance/design geometry determines the weights, rather than an arbitrary integer EDF. Signed weights and an independent Gaussian term are allowed by the Davies calculation. By default this function returns the upper tail, the convention commonly needed for a p-value.\n\nDavies evaluates a characteristic-function inversion with an accuracy target and an integration budget. On a normal successful path ``lower_tail=True`` returns the CDF. Status 2 emits a round-off warning while retaining the computed probability. Other failed central cases warn and invoke Liu; failed noncentral cases return NaN.\n\n**Current fallback convention:** the wrapper calls ``liu2`` with its default upper tail even when ``lower_tail=True``. That fallback uses only weights and degrees of freedom, omitting ``sigz``. Use ``liu2(..., lower_tail=True)`` explicitly when a lower-tail central approximation is intended. Do not infer a successful exact evaluation from the absence of an exception or ignore warnings; this port does not attach trace/status attributes to its returned probabilities.\n\nThe calibration of a GAM term statistic additionally depends on the fitted covariance, rank and smoothing assumptions. A numerical tail probability alone does not establish an exact finite-sample hypothesis test. See :doc:`/statistical_reference/inference` and `mgcv psum.chisq <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/psum.chisq.html>`_.",
    ),
    "liu2": (
        r"c_k=\sum_jh_j\lambda_j^k,\qquad\kappa_k(Q)=2^{k-1}(k-1)!\,c_k\quad(\text{central components})",
        "Liu's approximation matches standardized moments of a central weighted quadratic form to a scaled noncentral chi-square reference. Here c1 is its mean, 2*c2 its variance, and c3/c2**1.5 and c4/c2**2 determine the skewness/kurtosis matching branch. A scaled chi-square with one common positive weight is a useful exact-reference check.\n\nThis helper accepts central components only: it has no noncentrality or Gaussian-noise argument. It is an approximation for the intended nonnegative-definite form, not Davies integration for arbitrary signed weights. The routine checks vector lengths but does not comprehensively validate all mathematical preconditions.\n\n**Boundary convention:** thresholds at or below zero retain an initialized value of 1 for either tail setting. Degenerate nonpositive second-moment sums also return 1. For a nonnegative quadratic form, this agrees with the usual upper tail at negative thresholds, but it is not the mathematical lower-tail CDF. The selected chi-square CDF/SF is applied only to positive thresholds.\n\nThe higher-order matching follows the Liu, Tang and Zhang approximation cited by `mgcv psum.chisq <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/psum.chisq.html>`_. Inspect approximation quality rather than treating it as an exact test distribution.",
    ),
    "DaviesAlgorithm": (
        r"F_Q(q)=\frac12-\frac1\pi\int_0^\infty\frac{\operatorname{Im}\{e^{-itq}\varphi_Q(t)\}}{t}\,dt",
        "For a nondegenerate continuous quadratic-form law, characteristic-function inversion gives its CDF. Davies chooses integration/truncation parameters using error bounds for the weighted independent chi-square components and optional independent normal term. This is numerical probability evaluation, not simulation or model fitting.\n\nThe integration method returns the lower tail plus diagnostic trace and a fault code; it has no automatic Liu fallback. Use :doc:`psum_chisq` for the convenience wrapper, or this class when raw numerical diagnostics are required. The Python integration port is sequential; reuse of one mutable integration object across concurrent calls is not an advertised interface.\n\nCheck both probability range and fault code. Failed paths can return a negative probability sentinel, including an early auxiliary-budget exit whose code can still be zero. Degenerate all-zero components with zero Gaussian term use the implementation's strict comparison at zero. Use nondegenerate inputs for the usual continuous-law CDF interpretation. The implementation follows `mgcv's Davies source <https://github.com/cran/mgcv/blob/master/src/davies.c>`_.",
    ),
}


def generate_helpers(root):
    from feature_reference import heading, tree, write
    from reference_examples import example_rst

    for name, obj in helper_specs().items():
        formula, theory = HELPER_THEORY[name]
        text = heading(name) + f".. math::\n\n   {formula}\n\n" + theory + "\n\n"
        module = (
            "nampy.gam.fit.selection"
            if name
            in {
                "optimizer_endpoint_diagnostics",
                "supports_smoothing_method",
                "resolve_smoothing_method",
                "n_free_smoothing_params",
            }
            else obj.__module__
        )
        text += f"Import: ``from {module} import {name}``. These are backend helper APIs; core-model methods remain the usual analysis interface.\n\n"
        params, outputs = helper_contracts(name)
        text += contract(obj, params, outputs)
        if name == "DaviesAlgorithm":
            text += heading("Integration method", "-")
            text += contract(
                obj().davies,
                [
                    (
                        "lb",
                        "array-like[float], length r",
                        "required",
                        "Weights of the chi-square terms; signed weights are permitted.",
                    ),
                    (
                        "nc",
                        "array-like[float], length r",
                        "required",
                        "Finite nonnegative noncentralities, with no implicit scalar repetition at this low level.",
                    ),
                    (
                        "n",
                        "array-like[int], length r",
                        "required",
                        "Nonnegative integer degrees of freedom. The method casts to integer dtype; supply integers rather than relying on truncation.",
                    ),
                    (
                        "r",
                        "int > 0",
                        "required",
                        "Number of chi-square components, matching all three array lengths.",
                    ),
                    (
                        "sigma",
                        "float >= 0",
                        "required",
                        "Standard-normal multiplier, not its variance.",
                    ),
                    (
                        "c_val",
                        "float",
                        "required",
                        "One threshold q; call separately for several thresholds.",
                    ),
                    (
                        "lim",
                        "int > 0",
                        "required",
                        "Integration-term budget, corresponding to nlim on the wrapper.",
                    ),
                    (
                        "acc",
                        "float > 0",
                        "required",
                        "Requested numerical probability accuracy, corresponding to tol on the wrapper.",
                    ),
                ],
                [
                    (
                        "probability",
                        "float, shape ()",
                        None,
                        "Lower-tail probability on a successful evaluation; negative sentinels are possible on failed paths.",
                    ),
                    (
                        "trace",
                        "list[float], length 7",
                        None,
                        "Zero-based entries: [0] absolute-value sum/error diagnostic; [1] total integration terms; [2] number of integrations; [3] main integration interval; [4] initial truncation point; [5] convergence-factor SD; [6] parameter-search cycle count. Failed paths can contain partial/zero entries.",
                    ),
                    (
                        "ifault",
                        "int, shape ()",
                        None,
                        "Local return codes: 0 nominal completion, 1 insufficient integration budget/accuracy, 2 possible round-off error, 3 invalid component parameters. Also check probability range as explained above. These values are numerical diagnostics, not hypothesis-test outcomes.",
                    ),
                ],
                spelling="solver.davies",
                level="~",
            )
        text += example_rst(
            root, "extension_reference/statistical/" + name, helper_example(name)
        )
        write(root, "extension_reference/statistical/" + name + ".rst", text)
    write(
        root,
        "extension_reference/statistical/index.rst",
        tree(
            "Statistical helper APIs",
            list(helper_specs()),
            "These helpers expose fitting capabilities and endpoint diagnostics, prediction-grid masking, and the quadratic-form probabilities used by inference. Each page includes its theory, live signature, complete parameter/output contract and a standalone example. They operate on core GAM objects or numerical inputs; they do not add new estimator classes.",
        ),
    )


def generate(root):
    generate_links(root)
    generate_helpers(root)


def inventory_features():
    from nampy.gam.families.registry import make_gam_family

    return [
        (
            "link:" + name,
            make_gam_family,
            "statistical_reference/links/" + name,
            "statistical_link",
        )
        for name in LINKS
    ] + [
        (name, obj, "extension_reference/statistical/" + name, "statistical_helper")
        for name, obj in helper_specs().items()
    ]
