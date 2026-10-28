"""Family-specific explanations in the parameterization of each backend."""

from inspect import cleandoc

from reference_theory import block

FAMILIES = {
    "normal": block(
        """A Gaussian model describes symmetric variation around a conditional
        location. Its negative log density separates standardized squared error
        from the cost of assigning a large scale. With constant scale and an
        identity mean link, fitting the mean reduces to penalized least squares.""",
        r"-\log p(y\mid\mu,\sigma)=\tfrac12\log(2\pi)+\log\sigma+\frac{(y-\mu)^2}{2\sigma^2}.",
        """A location-scale model lets both location and scale depend on inputs:
        it can represent heteroscedasticity even though the mean remains
        additive on its own link scale. Sigma is conditional response spread,
        not a standard error of the fitted mean. Increasing scale reduces the
        residual term but increases log(sigma), preventing scale inflation from
        being a free improvement. Tail behavior remains Gaussian.""",
    ),
    "poisson": block(
        """Poisson regression models nonnegative event counts with equal
        conditional mean and variance. A log link makes the rate positive and
        gives an additive effect a multiplicative interpretation.""",
        r"\log\mu=\eta,\qquad -\log p(y\mid\mu)=\mu-y\log\mu+\log\Gamma(y+1).",
        """A one-unit predictor change multiplies the mean by exp(1); a term
        difference delta multiplies it by exp(delta). For varying exposure t,
        offset(log(t)) specifies expected count t*exp(eta_without_offset).
        Conditional overdispersion suggests checking missing predictors,
        dependence, or a different count law. Fractional nonnegative responses
        are not literal Poisson event counts.""",
    ),
    "binomial": block(
        """Binomial regression models a success probability from a specified
        number of trials. Bernoulli data are the one-trial case. A logit
        predictor is a log odds, so a term difference is a conditional odds
        ratio after exponentiation.""",
        r"\log\frac{p}{1-p}=\eta,\qquad -\ell=-y\log p-(m-y)\log(1-p)+\text{constant}.",
        """Trial totals carry information: a proportion from 100 trials is
        not equivalent to the same proportion from one trial. Use the documented
        grouped-response or trial-weight encoding. Odds ratios are not risk
        ratios, and probability changes depend on the starting probability.
        Large fitted log odds can indicate separation; smoothing regularizes
        penalized directions but does not guarantee every unpenalized
        coefficient is estimable.""",
    ),
    "gamma": block(
        """A Gamma response is positive and right-skewed. In mean/dispersion
        notation its shape is 1/phi and its rate is 1/(mu*phi). This makes
        relative variability constant when phi is constant.""",
        r"\operatorname{Var}(Y)=\phi\mu^2,\qquad \operatorname{CV}(Y)=\sqrt\phi,\qquad g(\mu)=\eta.",
        """With a log mean link, term differences multiply the conditional
        mean. A varying dispersion predictor changes relative variability
        independently of the mean. Zeros are outside the Gamma density; an
        exposure model or semicontinuous distribution is needed when zeros
        are genuine outcomes. The shape/rate neural interface and the
        mean/dispersion statistical interface require the parameter conversion
        above; the runtime output table determines the public column order.""",
    ),
    "negativebinom": block(
        """The negative binomial can be viewed as a Poisson law with a
        Gamma-distributed latent rate. Integrating out that rate gives
        overdispersed counts with a quadratic mean-variance relationship.""",
        r"\Lambda\sim\operatorname{Gamma}(r,\text{rate}=r/\mu),\quad Y\mid\Lambda\sim\operatorname{Poisson}(\Lambda),\quad \operatorname{Var}(Y)=\mu+\mu^2/r.",
        """Writing alpha=1/r gives variance mu+alpha*mu**2. Larger alpha
        increases overdispersion; r tending to infinity gives the Poisson
        limit. Distinguish a fixed nuisance dispersion, an estimated global
        dispersion, and a separate dispersion predictor: they are different
        models even if they use the same probability law. The model's mean
        remains mu; its spread is not sqrt(mu) as in a Poisson model.""",
    ),
    "beta": block(
        """Beta regression models a continuous proportion strictly between
        zero and one. Replacing shape parameters a,b by mean mu and precision
        phi separates the center of the distribution from its concentration.""",
        r"a=\mu\phi,\quad b=(1-\mu)\phi,\qquad E[Y]=\mu,\quad \operatorname{Var}(Y)=\frac{\mu(1-\mu)}{1+\phi}.",
        """A logit mean predictor keeps mu in (0,1); a positive precision
        transform keeps phi positive. Larger precision narrows the distribution
        at fixed mean. This is not a binomial proportion with a known trial
        count: phi models continuous dispersion, not sample size. Exact zeros
        and ones are outside the density; numerical clipping is a boundary
        convention and does not create point masses at either endpoint.""",
    ),
    "studentt": block(
        """A Student-t location-scale model replaces Gaussian quadratic tail
        growth with a logarithmic residual term. It can accommodate occasional
        large residuals while retaining a smooth location predictor.""",
        r"-\ell=\log\sigma+\frac{\nu+1}{2}\log\left(1+\frac{(y-\mu)^2}{\nu\sigma^2}\right)+c(\nu).",
        """Degrees of freedom nu control tail thickness; the Gaussian limit
        is nu tending to infinity. Sigma is a t scale, with response variance
        sigma**2*nu/(nu-2) when nu>2. Finite-variance student_lss fixes nu>2;
        scat and the neural Student-t interface have their own documented
        estimation/domain rules. Robustness to residual outliers does not
        automatically protect against influential covariate configurations.""",
    ),
    "inverse_gaussian": block(
        """The inverse-Gaussian law models positive skewed responses whose
        variance increases cubically with their mean. It arises, for example,
        as a first-passage-time distribution for drifted Brownian motion.""",
        r"\operatorname{Var}(Y)=\phi\mu^3,\qquad -\ell=\frac{(y-\mu)^2}{2\phi\mu^2y}+\frac12\log(2\pi\phi y^3).",
        """The mean link controls the conditional center; phi controls
        dispersion. Its cubic variance growth differs materially from Gamma's
        quadratic growth. A log mean link gives multiplicative mean effects,
        while other supported links change that interpretation. The response
        must be positive, and scale uncertainty is distinct from uncertainty
        in the estimated smooth.""",
    ),
    "lognormal": block(
        """A lognormal model assumes Gaussian variation after taking logs of
        the response. This is a response distribution, rather than merely
        fitting a positive mean through a log link.""",
        r"\log Y\sim N(m,s^2),\qquad \operatorname{median}(Y)=e^m,\quad E[Y]=e^{m+s^2/2}.",
        """The location parameter m is the mean of log(Y); exponentiating it
        gives the response median, not its mean. A varying s changes both
        response spread and response mean even if m is fixed. The density
        includes the 1/y transformation Jacobian. Compare effects on the
        intended scale: a location contribution is additive in log space,
        while natural response moments combine location and scale.""",
    ),
    "inversegamma": block(
        """If X is Gamma with shape a and rate r, then Y=1/X follows the
        inverse-Gamma law. This puts a long right tail on a positive response
        and makes existence of moments depend on the shape.""",
        r"-\log p(y)=-(a\log r-\log\Gamma(a))+(a+1)\log y+r/y.",
        """A valid density only requires positive a,r, but a finite mean
        requires a>1 and a finite variance requires a>2. Therefore a finite
        loss can coexist with an infinite natural moment. The neural positive
        parameter transform alone does not enforce these stronger moment
        conditions. Interpret point/spread outputs according to the moment
        domains rather than treating the rate parameter as response spread.""",
    ),
    "weibull": block(
        """The Weibull law models a positive response through scale lambda
        and shape k. Its survival and hazard functions explain the role of
        shape more directly than the density alone.""",
        r"S(y)=\exp\{-(y/\lambda)^k\},\qquad h(y)=\frac{k}{\lambda}(y/\lambda)^{k-1}.",
        """For k<1 the hazard decreases, k=1 gives an exponential law, and
        k>1 gives increasing hazard. Lambda rescales response time but is not
        generally the mean. A scale contribution and a shape contribution
        therefore have different interpretations. This response-density
        interface consumes observed positive values; it does not by itself
        add a censoring likelihood or survival-data encoding.""",
    ),
    "loglogistic": block(
        """A log-logistic response has a logistic distribution on the log
        scale. Its scale alpha is the response median, and beta controls the
        heaviness of its right tail.""",
        r"\log Y\sim\operatorname{Logistic}(\log\alpha,1/\beta),\qquad P(Y>y)=\{1+(y/\alpha)^\beta\}^{-1}.",
        """Mean and variance exist only for beta>1 and beta>2 respectively.
        Positive shape alone is sufficient for a density but not those
        moments. A larger alpha moves the response distribution upward;
        changing beta alters its tail and concentration. As with Weibull,
        using a positive response law is distinct from implementing censored
        survival fitting.""",
    ),
    "tweedie": block(
        """For power 1<p<2, a Tweedie response is a compound Poisson sum of
        positive Gamma jumps. The event count can be zero, creating an exact
        zero mass alongside a continuous positive density.""",
        r"Y=\sum_{j=1}^{N}Z_j,\qquad N\sim\operatorname{Poisson}(\lambda_N),\quad P(Y=0)=e^{-\lambda_N},\quad \operatorname{Var}(Y)=\phi\mu^p.",
        """The zero probability is coupled to mean, dispersion and power; it
        is not an independent zero-inflation predictor. p controls the
        mean-variance relationship, while phi controls its magnitude.
        Estimating power can require numerical density-series evaluation.
        Statistical tw can estimate a global power, twlss gives power and
        dispersion their own predictors, and the neural interface fixes the
        configured power. Their fitting routes are different even though
        the compound-law interpretation is shared.""",
    ),
    "categorical": block(
        """A categorical model assigns probability to each unordered label.
        Reference-category logits identify the statistical multinomial model
        by fixing category zero's logit, removing an arbitrary common shift.""",
        r"p_0=\frac{1}{1+\sum_{k=1}^{K}e^{\eta_k}},\qquad p_j=\frac{e^{\eta_j}}{1+\sum_{k=1}^{K}e^{\eta_k}},\qquad -\ell=-\log p_y.",
        """Each eta_j is log odds relative to category zero, not an
        independently fitted probability. All probabilities depend on all
        logits through the shared denominator. A neural categorical head
        instead supplies C logits and applies softmax; common logit shifts
        leave probabilities unchanged. The class-label ordering controls
        output columns and carries no numerical distance interpretation.""",
    ),
    "ordinal": block(
        """An ordinal model uses ordered thresholds to partition a latent
        location. A common predictor moves the cumulative probabilities
        together, while ordered cutpoints prevent negative class probabilities.""",
        r"P(Y\leq j)=F(c_j-\eta),\qquad P(Y=j)=F(c_j-\eta)-F(c_{j-1}-\eta).",
        """F is logistic for the proportional-odds construction. Threshold
        locations and an intercept are not independently identifiable without
        the constructor's anchoring rule. A larger eta shifts probability
        toward higher categories. This uses ordering without assuming equal
        distances between labels; moments of integer encodings depend on the
        chosen encoding. Class probabilities are often the most useful output.""",
    ),
    "dirichlet": block(
        """A Dirichlet response is a vector of positive fractions summing to
        one. Shape parameters can be split into a composition m and total
        concentration A, exposing the location/concentration tradeoff.""",
        r"A=\sum_j\alpha_j,\quad m_j=\alpha_j/A,\qquad \operatorname{Var}(Y_j)=\frac{m_j(1-m_j)}{A+1}.",
        """Increasing all alpha values proportionally leaves the mean
        composition unchanged and reduces variability. Components have
        negative conditional covariance because the row sum is fixed.
        This is a distribution over compositions, not a categorical
        observation or unconstrained multi-output Gaussian response. Zeros
        lie on the boundary and require the documented numerical handling
        or a model specifically designed for boundary mass.""",
    ),
    "mvnormdiag": block(
        """A diagonal multivariate Gaussian predicts one location and one
        scale per response dimension. Conditional independence makes the
        joint log density the sum of its scalar Gaussian log densities.""",
        r"\ell(y)=\sum_{j=1}^{D}\log N(y_j\mid\mu_j,\sigma_j^2),\qquad \operatorname{Cov}(Y\mid x)=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_D^2).",
        """Shared predictors can produce marginal association across
        outcomes, but residual conditional cross-covariance is fixed to zero.
        A wide output head therefore does not imply a full covariance model.
        Each location/scale pair has its own domain and contribution columns;
        the output contract fixes their concatenation order.""",
    ),
    "quantile": block(
        """Pinball loss targets a conditional quantile without specifying a
        normalized response density. At level q, underprediction and
        overprediction receive different linear costs.""",
        r"\rho_q(u)=u\{q-I(u<0)\},\qquad Q_q(x)\in\arg\min_v E[\rho_q(Y-v)\mid x].",
        """Predicting multiple levels estimates several locations in the
        conditional distribution, not its entire density. A monotone output
        transform can prevent crossing between requested quantiles.
        Pinball loss cannot be interpreted as a likelihood-based AIC, and
        finite quantile predictions alone do not determine a response
        variance or a calibrated prediction distribution.""",
    ),
    "gumbls": block(
        """The Gumbel law describes an extreme-value location-scale family
        with asymmetric tails. Let z=(y-mu)/beta. Its standardized density
        makes clear that mu is a location parameter rather than the mean.""",
        r"-\ell=\log\beta+z+e^{-z},\qquad E[Y]=\mu+\gamma_E\beta.",
        """A scale predictor changes both spread and the gap between
        location and mean. The family describes maxima-type extremes under
        its stated sign convention. Location and scale contributions belong
        to distinct predictors; adding their link components into one
        scalar response contribution has no valid interpretation.""",
    ),
    "gevlss": block(
        """Generalized extreme-value regression adds a shape parameter to
        location and scale. Shape determines whether the tail has a finite
        endpoint, a Gumbel limit, or a heavy upper tail.""",
        r"t=1+\xi(y-\mu)/\sigma>0,\qquad F(y)=\exp\{-t^{-1/\xi}\}.",
        """Support depends on every predicted parameter, so coefficient
        steps must preserve valid likelihood evaluations. The continuous
        xi=0 limit is Gumbel. A finite mean requires xi<1 and a finite
        variance requires xi<1/2. Constructor link bounds can impose stronger
        restrictions. Smooth shape predictors require enough tail information;
        a finite fit is not by itself evidence of stable extrapolation.""",
    ),
    "shash": block(
        """The sinh-arcsinh law transforms a Gaussian latent variable to
        control skewness and tail weight separately from location and scale.
        In NAMpy's statistical parameterization, the delta scale factor is
        part of the transformation.""",
        r"Z=\sinh\!\left[\delta\,\operatorname{asinh}\!\left(\frac{Y-\mu}{\sigma\delta}\right)-\epsilon\right]\sim N(0,1).",
        """The response density multiplies the Gaussian latent density by
        the absolute Jacobian of this transformation. Epsilon changes
        asymmetry; delta changes tail weight. At epsilon=0 and delta=1 the
        law reduces to a Gaussian. With skewness present, mu is generally
        neither the response mean nor its median; the documented point
        prediction uses the transformed median. Parameter contributions
        therefore need a parameter-specific interpretation.""",
    ),
}

ALIASES = {
    "gaussian": "normal",
    "gaulss": "normal",
    "gammals": "gamma",
    "nb": "negativebinom",
    "negbin": "negativebinom",
    "nb_lss": "negativebinom",
    "betar": "beta",
    "beta_lss": "beta",
    "scat": "studentt",
    "student_lss": "studentt",
    "tw": "tweedie",
    "twlss": "tweedie",
    "multinom": "categorical",
    "ocat": "ordinal",
    "robustnormal": "normal",
}


def family_theory(name, backend, *, include_heading=True):
    key = ALIASES.get(name, name)
    if name in {"zip", "zinb", "ziplss", "hurdlepoisson", "hurdlenegativebinom"}:
        hurdle = name in {"ziplss", "hurdlepoisson", "hurdlenegativebinom"} or (
            name == "zip" and backend == "statistical"
        )
        if hurdle:
            text = block(
                """A hurdle model separates occurrence from the size of a
                positive count. The positive-count law is truncated at zero,
                so only the occurrence component can generate a zero.""",
                r"P(Y=0)=\pi,\qquad P(Y=y>0)=(1-\pi)\frac{g(y)}{1-g(0)}.",
                """This differs from adding an extra zero mass to an
                untruncated count law. Changing the count intensity changes
                the mean among positive observations; changing pi changes
                the chance of any positive observation. The unconditional
                mean combines both mechanisms. A presence predictor is
                1-pi on the probability scale, not the zero probability pi.""",
            )
            if name == "zip" and backend == "statistical":
                text += "Statistical zip couples occurrence to intensity through its nuisance parameters; the two mechanisms do not have independently specified smooth predictors.\n\n"
        else:
            text = block(
                """A zero-inflated model mixes a structural-zero state with
                an ordinary count law g. Zeros can arise from either state,
                which can make the two mechanisms difficult to identify.""",
                r"P(Y=0)=\pi+(1-\pi)g(0),\qquad P(Y=y>0)=(1-\pi)g(y).",
                """The response mean is (1-pi) times the base count mean.
                Extra zeros can also reflect overdispersion or missing
                covariates, so a zero-inflation component is a substantive
                modeling choice. In ZINB, both base dispersion and the
                mixing probability affect zero frequency; inspect their
                uncertainty rather than attributing every zero to one cause.""",
            )
    elif name in {"quasi", "quasipoisson", "quasibinomial"}:
        text = block(
            """Quasi-likelihood specifies a mean and variance function without
            a complete probability density. Its estimating equation weights
            residuals by the inverse of the assumed variance.""",
            r"\frac{\partial Q(y,\mu)}{\partial\mu}=\frac{y-\mu}{\phi V(\mu)}.",
            """Quasi-Poisson takes V(mu)=mu; quasi-binomial takes
            V(mu)=mu*(1-mu), with the documented trial convention. Phi can
            allow extra variation while retaining a mean-link model. This
            does not supply probabilities for new observations, so normalized
            likelihood comparisons and likelihood-based AIC are unavailable.
            Mean-model diagnostics still depend on an adequate variance
            relation and independence assumptions.""",
        )
    elif name in {"cnorm", "clog", "cpois", "bcg"}:
        text = block(
            """Censoring describes partial knowledge of a latent response.
            An interval contributes its probability mass, rather than the
            density at either endpoint. A right-censored value contributes a
            survival probability.""",
            r"\ell_i=\begin{cases}\log f(y_i)&\text{exact},\\\log\{F(u_i)-F(l_i)\}&\text{continuous interval},\\\log\{1-F(l_i)\}&\text{right censored}.\end{cases}",
            """For closed integer Poisson intervals use F(u)-F(l-1).
            Replacing a bound with an ordinary observed response biases the
            target. Censoring models the observation process while retaining
            the latent law's parameters and moments. Truncation, which
            conditions on inclusion in the sample, requires a different
            likelihood normalization. Follow the exact target encoding
            below, including its endpoint and one-sided conventions.""",
        )
        if name == "bcg":
            text += "The Box-Cox transformation additionally contributes its Jacobian for exact observations. Its Gaussian location is on the transformed scale; inverse-transforming that location is not generally the response mean.\n\n"
    else:
        text = FAMILIES[key]
    if backend == "statistical" and name == "gaulss":
        text += "The low-level gaulss second predictor represents inverse standard deviation through its logb convention. Convert through the public natural-parameter contract before treating that column as sigma.\n\n"
    if backend == "statistical" and name == "gammals":
        text += "The source-family predictors use log mean and log dispersion. Public location-scale outputs can use sigma=sqrt(phi); inspect the natural output columns rather than assigning sigma to a raw second column.\n\n"
    if name == "robustnormal":
        text += "With rob set, NAMpy applies its documented robust transform to the Gaussian log density. This changes the training objective and is not a newly normalized Gaussian likelihood; rob=None recovers ordinary Gaussian NLL.\n\n"
    if backend == "neural":
        text += "Neural training composes this response objective with the chosen architecture. Its raw outputs pass through the configured domain transforms; likelihood parameterization and network architecture are separate choices.\n\n"
    return (
        (
            "Theory and interpretation\n-------------------------\n\n"
            if include_heading
            else ""
        )
        + text
        + cleandoc("""See :doc:`/statistical_reference/families/theory` for likelihoods,
        links, weights and distributional predictors. The mathematical law
        does not imply support for every fitting or smoothing-selection route;
        the contracts below define the implemented combinations.""")
        + "\n\n"
    )
