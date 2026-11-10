"""Rendered family contracts, parameterization notes and fitting guidance."""

from family_math import display_definition, distribution_contract, math


def parameterization_notes(name, backend):
    special = ""
    if backend == 'statistical' and name == 'bccg_lss':
        special += (
            "* **Parameters.** Natural columns ``mu, sigma, nu`` use log, log and identity links. Positive responses require a Jacobian and truncated-normal normalizer. At ``nu=0`` the law is lognormal.\n"
            "* **Moments.** Compute the mean and median from the response law. The mean is infinite for ``-1 <= nu < 0``. Variance is ``NaN`` there and infinite for ``-2 <= nu < -1``. Finite quadrature has numerical limits, including negative-power variance below ``sigma=sqrt(machine epsilon)``.\n"
            "* **Fitting.** Supply at least three distinct positive-weight responses. Formula estimability also depends on the fitted design.\n\n"
        )
    if backend == "statistical" and name == "beinf_lss":
        special += (
            "* **Predictors.** Formula keys ``mu, phi, pi0, pi1`` map to ``logit(mu)``, ``log(phi)``, ``log(pi0/pi_interior)`` and ``log(pi1/pi_interior)``. The atom predictors are coupled category log ratios.\n"
            "* **Natural output.** All five natural-parameter standard errors use the full covariance. The columns are ``mu, phi, pi0, pi1, pi_interior``. The fifth column depends on both atom predictors and has no separate additive term.\n"
            "* **Fitting.** All components and distinct interior observations are required. Neural fitting uses its normalized likelihood.\n"
            "* **R mapping.** R BEINF uses ``sigma=1/sqrt(1+phi)``, ``nu=pi0/pi_interior`` and ``tau=pi1/pi_interior``. NAMpy uses the normalized mixture variance. The saved gamlss.dist 6.1.1 variance hook adds the squared mean where that variance subtracts it.\n\n"
        )
    if backend == "statistical" and name == "zaga_lss":
        special += "Natural columns are ``mu, sigma, pi0`` with log/log/logit links. Mu is the positive Gamma mean, sigma its coefficient of variation, and pi0 the exact zero probability (gamlss.dist ZAGA calls it nu). The whole-response mean is (1-pi0)*mu. Log probability reports mass at zero and density elsewhere. Finite-logit fitting requires both components and distinct positive observations; exact zero/one atom probabilities are available only for probability queries. Neural ZAGA fitting uses its normalized likelihood.\n\n"
    if backend == "statistical" and name == "betabinomial_lss":
        special += "Natural columns are ``mu, kappa``, with logit/log links. Supply ``context={'trials': trials}`` at fit and for every explicit prediction dataset. Omitted X reuses fitted trials. The response is integer successes, not proportions or success/failure pairs; the mean count is trials*mu. R BB uses sigma=1/kappa and bd=trials. Fixed smoothing and automatic joint REML with Newton, BFGS and EFS are validated; neural fitting uses its normalized likelihood.\n\n"
    if backend == "statistical" and name == "lognormal_lss":
        special += "Natural columns are ``loc, scale`` for log(Y); identity/log predictors are used. loc may be negative. The density includes the response Jacobian -log(y); the response mean is exp(loc+scale**2/2), not loc or exp(loc). No LOGNO2/LNO parameterization or alternate links are implied.\n\n"
    if backend == "statistical" and name == "weibull_lss":
        special += "Natural columns are ``scale, shape``, both log-linked. The WEI mapping is mu=scale, sigma=shape; this is not WEI2 or WEI3. The response mean is scale*Gamma(1+1/shape).\n\n"
    if name in {"zip", "zinb", "ziplss", "hurdlepoisson", "hurdlenegativebinom"}:
        special += (
            "Here "
            + math("m")
            + " and "
            + math("v")
            + " denote the base count mean and variance.\n\n"
        )
        if backend == "statistical":
            special += "Statistical ``zip`` couples presence to log intensity through its nuisance coefficients; it does not estimate an independent zero-probability predictor. Statistical ``ziplss`` uses its documented presence link, so convert presence to zero probability before using this formula.\n\n"
    if name in {"quasi", "quasipoisson", "quasibinomial"}:
        special += "A likelihood-based AIC is unavailable for a quasi-likelihood family. Deviance and Pearson residuals retain their estimating-equation meanings.\n\n"
    if name in {"cnorm", "clog", "cpois", "bcg"}:
        special += "Targets include censoring bounds rather than treating bounds as ordinary observations. The worked example and parameterization notes below define the exact two-column encoding and one-sided sentinel convention.\n\n"
    if name == "cpois":
        special += (
            "For a closed integer interval "
            + math("[l,u]")
            + ", its probability is "
            + math("F(u)-F(l-1)")
            + "; continuous endpoint formulas must not be applied to integer intervals without the discrete correction.\n\n"
        )
        special += "Concentrated rare events with probability below machine epsilon use conditional-count cumulants for stable deviance derivatives with respect to the mean. These derivatives are checked against a 100-digit Poisson calculation: the upstream double-precision likelihood-ratio expressions can lose several digits through cancellation in extreme tails. Ordinary events retain the source expressions.\n\n"
    if name == "bcg":
        special += (
            "Box-Cox response moments require an inverse-transform calculation on its valid domain and need not exist for all negative-shape tails. NAMpy preserves the upstream transformed-response likelihood and returns the transformed Gaussian location; it does not return an inverse-transform response mean. Prior weight "
            + math("w")
            + " rescales transformed variance to "
            + math(r"\sigma^2/w")
            + ".\n\n"
        )
    if backend == "statistical" and name == "student_lss":
        special += "The gamlss.dist TF mapping uses the same location, t scale and degrees of freedom. Shared tail and likelihood theory is on :doc:`/statistical_reference/families/theory`.\n\n"
    if backend == "statistical" and name == "beta_lss":
        special += (
            "Alternative shape parameters are "
            + math(r"a=\mu\phi,\ b=(1-\mu)\phi")
            + ". For gamlss.dist BE, use "
            + math(r"\sigma_{\mathrm{BE}}=1/\sqrt{\phi+1}")
            + "; this is not NAMpy precision. Raw predictors are "
            + math(r"(\operatorname{logit}\mu,\log\phi)")
            + ".\n\n"
        )
    if backend == "statistical" and name == "nb_lss":
        special += (
            "The gamlss.dist NBI mapping is "
            + math(r"\mu_{\mathrm{NBI}}=\mu,\ \sigma_{\mathrm{NBI}}=\alpha")
            + ". In the shape/probability form use "
            + math(r"r=1/\alpha")
            + " and success probability "
            + math(r"r/(r+\mu)")
            + ". Raw predictors are "
            + math(r"(\log\mu,\log\alpha)")
            + ".\n\n"
        )
    return special


def mathematical_contract(name, backend):
    from family_theory import family_theory
    from feature_reference import heading

    definition, domains, mean, variance = distribution_contract(name, backend)
    formula, definition_notes = display_definition(name, definition)
    special = parameterization_notes(name, backend)
    if backend == "statistical":
        if name in {"gamma", "gammals"}:
            special += (
                "Statistical Gamma uses mean and dispersion: "
                + math(r"a=1/\phi,\ r=1/(\mu\phi)")
                + ". Public GAMLSS uses "
                + math(r"\sigma=\sqrt\phi")
                + ", so "
                + math(r"\operatorname{Var}(Y\mid x)=(\mu\sigma)^2")
                + ".\n\n"
            )
        if name == "betar":
            special += (
                "Mean/precision form: "
                + math(r"a=\mu\phi,\ b=(1-\mu)\phi")
                + ". The public beta_lss columns are "
                + math(r"(\mu,\phi)")
                + "; betar estimates/fixes precision as a nuisance parameter.\n\n"
            )
        if name == "scat":
            special += "Public location and scale do not imply Gaussian variance. student_lss defaults to fixed df=5; scat has its own nuisance-parameter estimation convention.\n\n"
    if backend == "neural" and name in {"negativebinom", "zinb", "hurdlenegativebinom"}:
        special += (
            "Neural dispersion is inverse total count: "
            + math(r"r=1/\alpha")
            + "; "
            + math(r"\operatorname{Var}(Y_{\mathrm{base}}\mid x)=\mu+\alpha\mu^2")
            + ".\n\n"
        )
    if name == "shash":
        special += (
            "The public point prediction is the conditional median "
            + math(r"\mu+\sigma\delta\sinh(\epsilon/\delta)")
            + "; location "
            + math(r"\mu")
            + " is not generally the mean or median. The moments above use the mgcv scale reparameterization including the "
            + math(r"\delta")
            + " factor.\n\n"
        )
    if name == "robustnormal":
        special += (
            "The underlying response law is Gaussian. If ``rob`` is not None, its log density "
            + math(r"\ell")
            + " is replaced by "
            + math(
                r"\operatorname{softplus}(\ell+\mathrm{rob})-\operatorname{softplus}(\mathrm{rob})"
            )
            + "; training minimizes its negative. This transformed objective is not a normalized likelihood. ``rob=None`` recovers ordinary NLL.\n\n"
        )
    text = (
        f".. math::\n\n   {formula}\n\n"
        + "".join(
            f"* **{label}:** {value}\n"
            for label, value in [
                ("Natural domains", domains),
                ("Conditional mean", mean),
                ("Conditional variance", variance),
            ]
        )
        + "\n"
        + (definition_notes + "\n\n" if definition_notes else "")
        + special
    )
    if backend == "statistical" and name in {"bccg_lss", "student_lss", "beta_lss", "nb_lss", "lognormal_lss", "weibull_lss", "betabinomial_lss", "zaga_lss", "beinf_lss"}:
        reference = {"bccg_lss": "BCCG", "student_lss": "TF", "beta_lss": "BE", "nb_lss": "NBI", "lognormal_lss": "LOGNO", "weibull_lss": "WEI", "betabinomial_lss": "BB", "zaga_lss": "ZAGA", "beinf_lss": "BEINF"}[name]
        text += f"Distribution reference: `gamlss.dist::{reference} <https://github.com/cran/gamlss.dist/blob/master/man/{reference}.Rd>`_. Fitting uses NAMpy's joint penalized engine, not gamlss RS backfitting.\n\n"
        if name == 'bccg_lss':
            text += (
                "* **Fit.** Fixed smoothing and joint REML with ``outer_newton``, ``bfgs`` or ``efs`` are validated, including fits with three smoothed predictors. General-family method aliases resolve to REML. ``optim``, ``magic`` and ``nlm.fd`` are unavailable.\n"
                "* **Numerical checks.** Derivatives through order four include the normalizer and zero-power limit. Outer checks cover offsets and nonunit or zero weights. An independent derivative-free REML profile checks selected endpoints.\n"
                "* **Inference and limits.** Conditional and approximate smoothing-adjusted covariance are available. Simulation or refit centile bands, neural fitting, alternate links, censoring, truncation and deviance residuals are outside this route. Use the separate censored ``bcg`` likelihood for its own supported scope. Fits that stop at a nonconverged float64 plateau are ineligible for family ranking.\n\n"
            )
        if name in {"lognormal_lss", "weibull_lss", "betabinomial_lss", "zaga_lss", "beinf_lss"}:
            text += (
                "* **Fit.** Use fixed smoothing or automatic general-family REML with ``outer_newton``, ``bfgs`` or ``efs``. Method labels follow mgcv's coercion to REML. The optim route is unavailable because it requires null-deviance scaling. ``magic`` and ``nlm.fd`` are also unsupported.\n"
                "* **Diagnostics.** Use shared distribution quantile residuals. Native deviance residuals and null deviance are unavailable.\n"
                "* **Scope.** The fitting route excludes censoring, truncation, alternate links, and guarded transformed-coefficient, transformed-observation and general NCV configurations.\n\n"
            )
    elif backend == "statistical":
        text += "References: `mgcv distribution-family definitions <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/family.mgcv.html>`_. NAMpy fitting support is listed below.\n\n"
        if name in {"gammals", "shash", "bcg"}:
            text += f"`Upstream {name} parameterization <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/{name}.html>`_.\n\n"
    else:
        text += "References: `PyTorch probability distributions <https://docs.pytorch.org/docs/stable/distributions.html>`_. NAMpy defines its family transforms and point outputs below.\n\n"

    if not (backend == "statistical" and name == "student_lss"):
        text += family_theory(name, backend, include_heading=False)
    if backend == "neural":
        if name == "normal_observation":
            text += "Use ObservationTarget for fitting/scoring. The internal finite batching encoding has columns (value, upper, code); prediction parameters describe the latent normal while probability queries condition on truncation.\n\n"
        if name == "beinf":
            text += "Four raw predictors produce five natural columns: mu, phi, pi0, pi1, pi_interior. No fifth additive predictor is padded.\n\n"
        if name == "betabinomial":
            text += "Supply distributional_kwargs={'trials': m}, a fixed integer m>=2. Row-specific trial counts are unsupported by this neural route.\n\n"
        if name == "categorical":
            text += "The low-level categorical loss also accepts one-hot/probability targets of shape ``(n,C)`` (or ``(n,1,C)``), reducing them by argmax rather than fitting a soft-label cross-entropy objective. Scalar labels may have shape ``(n,1)``.\n\n"
        target = (
            "(n,D) floating, positive rows summing to one"
            if name == "dirichlet"
            else "(n,D) floating response matrix"
            if name == "mvnormdiag"
            else "(n,) integer class indices 0,...,C-1 (estimator encodes original labels)"
            if name in {"categorical", "ordinal"}
            else "(n,) floating scalar response; support follows the distribution definition above"
        )
        text += (
            heading("Targets, transforms and scoring", "-")
            + f"Target shape/encoding: ``{target}``. Network rows must align with target rows. Parameter column order is given in the output contract below. Positive default transforms use ``softplus(raw)+eps``; identity leaves unrestricted locations unchanged. Constructor transform options accept registered transform names or callables and must preserve the stated parameter domains.\n\n"
        )
        text += "``compute_loss`` and ``compute_loss_per_sample`` consume raw network outputs, not already transformed parameters. The scalar loss is averaged over observations. Lower loss is better. LSS scoring follows the shared estimator contract; response SD summarizes conditional variability, not estimation uncertainty. Distribution-specific moments that do not exist cannot be interpreted as finite uncertainty estimates.\n\n"
    else:
        text += "Target encoding, response/link column meanings and supported fitting criteria are specified in the tables and examples below. Scalar uncensored targets have shape ``(n,)``. Residual choices are family-specific; use :doc:`/statistical_reference/methods/residuals`. Log likelihood is higher-is-better; AIC/BIC are lower-is-better and require a normalized comparable likelihood.\n\n"
    text += (
        "Implementation definitions: "
        + (
            ":mod:`nampy.neural.families`"
            if backend == "neural"
            else ":mod:`nampy.gam.families`"
        )
        + ". See the source link on the runtime contract for the implemented transforms and objective.\n\n"
    )
    return text
