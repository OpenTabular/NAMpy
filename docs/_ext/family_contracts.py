"""Rendered family contracts, parameterization notes and fitting guidance."""

from family_math import display_definition, distribution_contract, math


def parameterization_notes(name, backend):
    special = ""
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
    if backend == "statistical" and name in {"student_lss", "beta_lss", "nb_lss"}:
        reference = {"student_lss": "TF", "beta_lss": "BE", "nb_lss": "NBI"}[name]
        text += f"Distribution reference: `gamlss.dist::{reference} <https://github.com/cran/gamlss.dist/blob/master/man/{reference}.Rd>`_. Fitting uses NAMpy's joint penalized engine, not gamlss RS backfitting. See :doc:`/development/neighbor_parity` for the tested mapping and numerical comparisons.\n\n"
    elif backend == "statistical":
        text += "References: `mgcv distribution-family definitions <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/family.mgcv.html>`_. NAMpy fitting support is listed below.\n\n"
        if name in {"gammals", "shash", "bcg"}:
            text += f"`Upstream {name} parameterization <https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/{name}.html>`_.\n\n"
    else:
        text += "References: `PyTorch probability distributions <https://docs.pytorch.org/docs/stable/distributions.html>`_. NAMpy defines its family transforms and point outputs below.\n\n"

    if not (backend == "statistical" and name == "student_lss"):
        text += family_theory(name, backend, include_heading=False)
    if backend == "neural":
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
            ":mod:`nampy.neural.distributions.distributions`"
            if backend == "neural"
            else ":mod:`nampy.gam.families`"
        )
        + ". See the source link on the runtime contract for the implemented transforms and objective.\n\n"
    )
    return text
