"""Mathematical contracts shared only when the parameterizations agree.

Notation: phi is dispersion, alpha/beta positive shapes, r a positive rate,
sigma a positive standard deviation or the explicitly named scale.
"""

# Definition, natural order/domains, conditional mean, conditional variance.
DISTRIBUTIONS = {
    "normal": (
        r"Y=mu+sigma Z, Z~Normal(0,1); f(y)=exp(-(y-mu)^2/(2 sigma^2))/(sqrt(2 pi) sigma).",
        "mu in R; sigma>0",
        "mu",
        "sigma^2",
    ),
    "poisson": (
        r"P(Y=y)=exp(-lambda) lambda^y/y!, y=0,1,... .",
        "lambda>0",
        "lambda",
        "lambda",
    ),
    "gamma": (
        r"f(y)=r^a y^(a-1) exp(-r y)/Gamma(a), y>0.",
        "shape a>0; rate r>0 (inverse scale)",
        "a/r",
        "a/r^2",
    ),
    "beta": (
        r"f(y)=y^(alpha-1)(1-y)^(beta-1)/B(alpha,beta), 0<y<1.",
        "alpha>0; beta>0",
        "alpha/(alpha+beta)",
        "alpha beta/((alpha+beta)^2 (alpha+beta+1))",
    ),
    "studentt": (
        r"Y=loc+scale T_df, T_df has standard Student-t density proportional to (1+t^2/df)^(-(df+1)/2).",
        "df>0; loc in R; scale>0",
        "loc for df>1; undefined otherwise",
        "scale^2 df/(df-2) for df>2; infinite for 1<df<=2; undefined for df<=1",
    ),
    "negativebinom": (
        r"P(Y=y)=Gamma(y+r)/(Gamma(r)y!) (r/(r+mu))^r (mu/(r+mu))^y.",
        "mu>0; total count r>0",
        "mu",
        "mu+mu^2/r",
    ),
    "inversegamma": (
        r"f(y)=r^a y^(-a-1) exp(-r/y)/Gamma(a), y>0.",
        "shape a>0; rate r>0 (scale in the inverse-gamma density)",
        "r/(a-1) for a>1; infinite otherwise",
        "r^2/((a-1)^2(a-2)) for a>2; infinite otherwise",
    ),
    "lognormal": (
        r"log(Y)~Normal(loc,scale^2), Y>0.",
        "loc in R; scale>0 on log-response scale",
        "exp(loc+scale^2/2)",
        "(exp(scale^2)-1) exp(2 loc+scale^2)",
    ),
    "weibull": (
        r"f(y)=(k/lambda)(y/lambda)^(k-1) exp(-(y/lambda)^k), y>0.",
        "scale lambda>0; shape k>0",
        "lambda Gamma(1+1/k)",
        "lambda^2 [Gamma(1+2/k)-Gamma(1+1/k)^2]",
    ),
    "loglogistic": (
        r"F(y)=1/(1+(y/alpha)^(-beta)), y>0.",
        "scale alpha>0; shape beta>0",
        "alpha (pi/beta)/sin(pi/beta) for beta>1; infinite otherwise",
        "alpha^2 [(2pi/beta)/sin(2pi/beta)-((pi/beta)/sin(pi/beta))^2] for beta>2; infinite otherwise",
    ),
    "categorical": (
        r"P(Y=j)=p_j, j=0,...,C-1; sum(p)=1.",
        "class probabilities p_j>=0, normalized across C classes",
        "Probability vector p; a numeric class-label mean has no invariant categorical interpretation",
        "Covariance of one-hot Y: diag(p)-p p^T",
    ),
    "ordinal": (
        r"P(Y<=j)=sigmoid(c_j-eta), with ordered cutpoints c_j; class probabilities are successive cumulative differences.",
        "eta in R; ordered fitted thresholds, C classes",
        "Class probabilities determine moments of encoded labels; predict_point chooses a class",
        "For integer labels: sum(j^2 p_j)-(sum(j p_j))^2",
    ),
    "dirichlet": (
        r"f(y)=Gamma(A)/prod Gamma(alpha_i) prod y_i^(alpha_i-1), A=sum(alpha_i); y_i>0, sum(y)=1.",
        "alpha_1,...,alpha_D>0",
        "alpha_i/A",
        "alpha_i(A-alpha_i)/(A^2(A+1)); off-diagonal covariance -alpha_i alpha_j/(A^2(A+1))",
    ),
    "mvnormdiag": (
        r"Y_i independently Normal(loc_i,scale_i^2) conditional on inputs; joint density is the product over dimensions.",
        "loc_1,...,loc_D in R, then scale_1,...,scale_D>0",
        "loc vector",
        "Diagonal covariance diag(scale_i^2)",
    ),
    "quantile": (
        r"For each level q, minimize rho_q(y-v)=(y-v)(q-I[y<v]); sum over levels and average over rows. This is a loss, not a normalized density.",
        "Requested quantiles in configured level order; levels in (0,1)",
        "Not determined by a finite quantile vector; point output uses the 0.5 quantile when present, otherwise the arithmetic average of the predicted quantiles",
        "Not determined; predict_spread is unavailable",
    ),
    "tweedie": (
        r"Compound Poisson-gamma law for 1<p<2: N~Poisson(mu^(2-p)/(phi(2-p))); Y is a sum of N Gamma((2-p)/(p-1), scale=phi(p-1)mu^(p-1)) jumps.",
        "mu>0; phi>0; fixed power 1<p<2",
        "mu",
        "phi mu^p",
    ),
    "gumbls": (
        r"F(y)=exp(-exp(-(y-mu)/beta)), y in R.",
        "location mu in R; scale beta>0",
        "mu+EulerGamma beta",
        "pi^2 beta^2/6",
    ),
    "gevlss": (
        r"F(y)=exp(-(1+xi (y-mu)/sigma)^(-1/xi)); support 1+xi(y-mu)/sigma>0. The xi=0 limit is Gumbel.",
        "mu in R; sigma>0; shape xi subject to the configured link bounds",
        "mu+sigma (Gamma(1-xi)-1)/xi for xi<1 (continuous xi=0 limit); undefined otherwise",
        "sigma^2 (Gamma(1-2xi)-Gamma(1-xi)^2)/xi^2 for xi<1/2; infinite otherwise",
    ),
    "shash": (
        r"Z=sinh(delta asinh((Y-mu)/(sigma delta))-epsilon)~Normal(0,1); density is the normal density of Z times the absolute transformation Jacobian.",
        "mu in R; sigma>0; skew epsilon in R; tail delta>0",
        "mu+sigma delta E[sinh((asinh(Z)+epsilon)/delta)], Z standard normal",
        "(sigma delta)^2 Var[sinh((asinh(Z)+epsilon)/delta)]; finite for delta>0",
    ),
}
ALIASES = {
    "gaussian": "normal",
    "gaulss": "normal",
    "student_lss": "studentt",
    "scat": "studentt",
    "nb": "negativebinom",
    "negbin": "negativebinom",
    "nb_lss": "negativebinom",
    "betar": "beta",
    "beta_lss": "beta",
    "tw": "tweedie",
    "twlss": "tweedie",
    "multinom": "categorical",
    "ocat": "ordinal",
    "robustnormal": "normal",
    "gammals": "gamma",
}


def mathematical_contract(name, backend):
    from family_theory import family_theory
    from feature_reference import heading

    key = ALIASES.get(name, name)
    special = ""
    if key in {"zip", "zinb", "ziplss", "hurdlepoisson", "hurdlenegativebinom"}:
        nb = key in {"zinb", "hurdlenegativebinom"}
        base = (
            "negative binomial with mean mu and total count r"
            if nb
            else "Poisson with rate lambda"
        )
        hurdle = (
            key.startswith("hurdle")
            or key == "ziplss"
            or (backend == "statistical" and key == "zip")
        )
        definition = f"Base count law g is {base}. "
        if hurdle:
            definition += "P(Y=0)=pi; P(Y=y>0)=(1-pi)g(y)/(1-g(0)). Set m_plus=m/(1-g(0)) and v_plus=(v+m^2)/(1-g(0))-m_plus^2."
            mean, variance = "(1-pi)m_plus", "(1-pi)v_plus+pi(1-pi)m_plus^2"
        else:
            definition += "P(Y=0)=pi+(1-pi)g(0); P(Y=y>0)=(1-pi)g(y)."
            mean, variance = "(1-pi)m", "(1-pi)v+pi(1-pi)m^2"
        contract = (
            definition,
            "zero probability pi in [0,1); positive base rate/mean; NB total count r>0",
            mean,
            variance,
        )
        special += "Here m and v denote the base count mean and variance.\n\n"
        if backend == "statistical":
            special += "Statistical ``zip`` couples presence to log intensity through its nuisance coefficients; it does not estimate an independent zero-probability predictor. Statistical ``ziplss`` uses its documented presence link, so convert presence to zero probability before using this formula.\n\n"
    elif key in {"quasi", "quasipoisson", "quasibinomial"}:
        variance = {
            "quasi": "phi V(mu), using the configured variance function",
            "quasipoisson": "phi mu",
            "quasibinomial": "phi mu(1-mu)",
        }[key]
        contract = (
            "Quasi-likelihood specifies a mean-variance relation rather than a normalized probability density.",
            "mu in the domain of the configured link; dispersion phi>0",
            "mu",
            variance,
        )
        special += "A likelihood-based AIC is unavailable for a quasi-likelihood family. Deviance and Pearson residuals retain their estimating-equation meanings.\n\n"
    elif key in {"cnorm", "clog", "cpois", "bcg"}:
        law = {
            "cnorm": "Normal",
            "clog": "logistic",
            "cpois": "Poisson",
            "bcg": "Box-Cox-transformed Gaussian",
        }[key]
        contract = (
            f"Latent response follows a {law} law. Exact observations contribute f(y); intervals contribute F(upper)-F(lower); left/right censoring contributes F(upper) or 1-F(lower).",
            "Latent-law location/mean and scale/transformation nuisance parameters; see the source-specific constructor and target encoding below",
            "Latent-law mean, not the mean of the censoring endpoint",
            "Latent-law variance, not the variance of recorded censoring endpoints",
        )
        special += "Targets include censoring bounds rather than treating bounds as ordinary observations. The worked example and parameterization notes below define the exact two-column encoding and one-sided sentinel convention.\n\n"
    elif key in {"binomial"}:
        contract = (
            "Y successes in m trials; P(Y=y)=choose(m,y)p^y(1-p)^(m-y). Bernoulli is m=1.",
            "p in [0,1]; positive trials m",
            "m p (p for proportions)",
            "m p(1-p) (p(1-p)/m for proportions)",
        )
    elif key == "inverse_gaussian":
        contract = (
            "f(y)=sqrt(lambda/(2 pi y^3)) exp(-lambda(y-mu)^2/(2 mu^2 y)), y>0.",
            "mu>0; precision lambda=1/phi>0",
            "mu",
            "phi mu^3",
        )
    else:
        if key not in DISTRIBUTIONS:
            raise ValueError(f"Missing mathematical contract: {backend}/{name}")
        contract = DISTRIBUTIONS[key]
    if name in {"cnorm", "clog", "cpois"}:
        base_definition, domains, mean, variance = {
            "cnorm": (
                "Latent CDF F(y)=Phi((y-mu)/sigma), y in R",
                "mu in R; sigma>0",
                "mu",
                "sigma^2",
            ),
            "clog": (
                "Latent CDF F(y)=sigmoid((y-mu)/s), y in R",
                "mu in R; logistic scale s>0",
                "mu",
                "pi^2 s^2/3",
            ),
            "cpois": (
                "Latent P(Y=j)=exp(-lambda)lambda^j/j!, j=0,1,...",
                "lambda>0",
                "lambda",
                "lambda",
            ),
        }[name]
        contract = (
            base_definition
            + ". Censoring contributions follow the documented exact/interval/one-sided encoding.",
            domains,
            mean,
            variance,
        )
        if name == "cpois":
            special += "For a closed integer interval [l,u], its probability is F(u)-F(l-1); continuous endpoint formulas must not be applied to integer intervals without the discrete correction.\n\n"
    if name == "bcg":
        contract = (
            "Z=h_lambda(Y) is Gaussian: h_lambda(y)=(y^lambda-1)/lambda for lambda!=0, and log(y) at lambda=0. The uncensored response density kernel is NormalPDF((h_lambda(y)-mu)/sigma)*y^(lambda-1)/sigma; censoring uses the transformed bounds in Gaussian CDF differences.",
            "Y>0; transformed location mu in R; sigma>0; Box-Cox lambda in R. Inverse h_lambda requires 1+lambda Z>0.",
            "E[Z]=mu at unit prior weight. Response E[Y]=E[h_lambda^(-1)(Z)]; for lambda=0 this is exp(mu+sigma^2/2).",
            "Var(Z)=sigma^2 at unit prior weight. Response Var(Y)=E[h_lambda^(-1)(Z)^2]-E[h_lambda^(-1)(Z)]^2; for lambda=0 this is (exp(sigma^2)-1)exp(2mu+sigma^2).",
        )
        special += "Box-Cox response moments require an inverse-transform calculation on its valid domain and need not exist for all negative-lambda tails. NAMpy preserves the upstream transformed-response likelihood and returns the transformed Gaussian location; it does not return an inverse-transform response mean. Prior weight w rescales transformed variance to sigma^2/w.\n\n"
    if backend == "statistical" and name == "student_lss":
        contract = (
            "Y=mu+sigma*T_df, with fixed degrees of freedom; NAMpy fits (mu,sigma) jointly.",
            "mu in R; sigma>0; fixed df must be finite and >2 (default 5)",
            "mu",
            "sigma^2*df/(df-2)",
        )
        special += "Broader mathematical note: a Student-t law is defined for df>0, but NAMpy's student_lss constructor restricts df>2 to ensure finite response variance. It fits no degrees-of-freedom predictor. The gamlss.dist TF mapping is nu=df, with the same mu and sigma (t scale).\n\n"
    elif backend == "statistical" and name == "beta_lss":
        contract = (
            "Y~Beta(mu*phi,(1-mu)*phi); natural columns are (mu,phi), mean and precision.",
            "0<mu<1; finite phi>0; target 0<Y<1",
            "mu",
            "mu*(1-mu)/(1+phi)",
        )
        special += "Alternative shape parameters are alpha=mu*phi and beta=(1-mu)*phi. For gamlss.dist BE, use sigma_BE=1/sqrt(phi+1); sigma_BE is not NAMpy precision phi. Raw predictors are (logit(mu),log(phi)).\n\n"
    elif backend == "statistical" and name == "nb_lss":
        contract = (
            "Negative-binomial counts with natural columns (mu,alpha), mean and inverse-shape dispersion; total_count=1/alpha.",
            "finite mu>0; finite alpha>0; target integers 0,1,...",
            "mu",
            "mu+alpha*mu^2",
        )
        special += "The gamlss.dist NBI mapping is mu_NBI=mu and sigma_NBI=alpha. In the shape/probability form use r=1/alpha and success probability r/(r+mu). Raw predictors are (log(mu),log(alpha)).\n\n"
    definition, domains, mean, variance = contract
    if backend == "statistical":
        if name in {"gamma", "gammals"}:
            special += "Statistical Gamma uses mean mu and dispersion phi: shape=1/phi and rate=1/(mu phi). Public GAMLSS uses sigma=sqrt(phi), so Var(Y)=(mu sigma)^2.\n\n"
        if name == "betar":
            special += "Mean/precision form: alpha=mu phi, beta=(1-mu)phi. The public beta_lss columns are (mu,phi); betar estimates/fixes precision as a nuisance parameter.\n\n"
        if name == "scat":
            special += "Public location and scale do not imply Gaussian variance. student_lss defaults to fixed df=5; scat has its own nuisance-parameter estimation convention.\n\n"
    if backend == "neural" and name in {"negativebinom", "zinb", "hurdlenegativebinom"}:
        special += "Neural dispersion alpha is inverse total count: r=1/alpha; Var(base count)=mu+alpha mu^2.\n\n"
    if name == "shash":
        special += "The public point prediction is the conditional median mu+sigma*delta*sinh(epsilon/delta); location mu is not generally the mean or median. The moments above use the mgcv scale reparameterization including the delta factor.\n\n"
    if name == "robustnormal":
        special += "The underlying response law is Gaussian. If rob is not None, its log density l is replaced by softplus(l+rob)-softplus(rob); training minimizes its negative. This transformed objective is not a normalized likelihood. rob=None recovers ordinary NLL.\n\n"
    text = (
        heading("Theory and interpretation", "-")
        + "".join(
            f"* **{label}:** {value}\n"
            for label, value in [
                ("Distribution / loss", definition),
                ("Natural domains", domains),
                ("Conditional mean", mean),
                ("Conditional variance", variance),
            ]
        )
        + "\n"
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
