"""Authored LaTeX for family laws, domains and conditional moments.

Keep prose outside math roles, and require explicit coverage for new families.
The statistical and neural interfaces share formulas only where their
parameterizations agree.
"""

import re


def math(expression):
    return f":math:`{expression}`"


M = math

NORMAL = (
    M(r"Y=\mu+\sigma Z,\quad Z\sim\mathcal N(0,1)")
    + "; "
    + M(r"f(y)=\frac{\exp\{-(y-\mu)^2/(2\sigma^2)\}}{\sqrt{2\pi}\,\sigma}"),
    M(r"\mu\in\mathbb R,\quad\sigma>0"),
    M(r"\mathbb E[Y\mid x]=\mu"),
    M(r"\operatorname{Var}(Y\mid x)=\sigma^2"),
)
POISSON = (
    M(r"\Pr(Y=y)=e^{-\lambda}\lambda^y/y!,\quad y\in\{0,1,\ldots\}"),
    M(r"\lambda>0"),
    M(r"\mathbb E[Y\mid x]=\lambda"),
    M(r"\operatorname{Var}(Y\mid x)=\lambda"),
)
STUDENT = (
    M(r"Y=\mu+\sigma T_\nu,\quad T_\nu\sim t_\nu")
    + "; "
    + M(
        r"f(y)=\frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\,\sigma}\left(1+\frac{(y-\mu)^2}{\nu\sigma^2}\right)^{-(\nu+1)/2}"
    ),
    M(r"\nu>0,\quad\mu\in\mathbb R,\quad\sigma>0")
    + "; "
    + M(r"\mu")
    + " is location and "
    + M(r"\sigma")
    + " is t scale.",
    M(r"\mathbb E[Y\mid x]=\mu") + " for " + M(r"\nu>1") + "; undefined otherwise.",
    M(r"\operatorname{Var}(Y\mid x)=\sigma^2\nu/(\nu-2)")
    + " for "
    + M(r"\nu>2")
    + "; infinite for "
    + M(r"1<\nu\leq2")
    + "; undefined for "
    + M(r"\nu\leq1")
    + ".",
)
NEGATIVE_BINOMIAL = (
    M(
        r"\Pr(Y=y)=\frac{\Gamma(y+r)}{\Gamma(r)y!}\left(\frac{r}{r+\mu}\right)^r\left(\frac{\mu}{r+\mu}\right)^y,\quad y\in\{0,1,\ldots\}"
    ),
    M(r"\mu>0,\quad r>0"),
    M(r"\mathbb E[Y\mid x]=\mu"),
    M(r"\operatorname{Var}(Y\mid x)=\mu+\mu^2/r"),
)
GAMMA = (
    M(r"f(y)=\frac{r^a}{\Gamma(a)}y^{a-1}e^{-ry},\quad y>0"),
    "Shape " + M(r"a>0") + "; rate " + M(r"r>0") + " (inverse scale).",
    M(r"\mathbb E[Y\mid x]=a/r"),
    M(r"\operatorname{Var}(Y\mid x)=a/r^2"),
)
BETA = (
    M(r"f(y)=\frac{y^{a-1}(1-y)^{b-1}}{\mathrm B(a,b)},\quad 0<y<1"),
    M(r"a>0,\quad b>0"),
    M(r"\mathbb E[Y\mid x]=\frac{a}{a+b}"),
    M(r"\operatorname{Var}(Y\mid x)=\frac{ab}{(a+b)^2(a+b+1)}"),
)
DISTRIBUTIONS = {
    "normal": NORMAL,
    "poisson": POISSON,
    "studentt": STUDENT,
    "negativebinom": NEGATIVE_BINOMIAL,
    "gamma": GAMMA,
    "beta": BETA,
    "binomial": (
        M(r"\Pr(Y=y)=\binom{m}{y}p^y(1-p)^{m-y},\quad y\in\{0,\ldots,m\}")
        + "; Bernoulli when "
        + M(r"m=1")
        + ".",
        M(r"p\in[0,1],\quad m\in\mathbb N_{>0}"),
        M(r"\mathbb E[Y\mid x]=mp,\quad\mathbb E[Y/m\mid x]=p"),
        M(
            r"\operatorname{Var}(Y\mid x)=mp(1-p),\quad\operatorname{Var}(Y/m\mid x)=p(1-p)/m"
        ),
    ),
    "inverse_gaussian": (
        M(
            r"f(y)=\sqrt{\frac{\lambda}{2\pi y^3}}\exp\left\{-\frac{\lambda(y-\mu)^2}{2\mu^2y}\right\},\quad y>0"
        ),
        M(r"\mu>0,\quad\lambda=1/\phi>0"),
        M(r"\mathbb E[Y\mid x]=\mu"),
        M(r"\operatorname{Var}(Y\mid x)=\phi\mu^3"),
    ),
    "inversegamma": (
        M(r"f(y)=\frac{r^a}{\Gamma(a)}y^{-a-1}e^{-r/y},\quad y>0"),
        M(r"a>0,\quad r>0")
        + "; "
        + M(r"r")
        + " is the scale in the inverse-Gamma density.",
        M(r"\mathbb E[Y\mid x]=r/(a-1)")
        + " for "
        + M(r"a>1")
        + "; infinite otherwise.",
        M(r"\operatorname{Var}(Y\mid x)=\frac{r^2}{(a-1)^2(a-2)}")
        + " for "
        + M(r"a>2")
        + "; no finite variance otherwise.",
    ),
    "lognormal": (
        M(r"\log Y\sim\mathcal N(\mu,\sigma^2),\quad Y>0"),
        "Log-response location "
        + M(r"\mu\in\mathbb R")
        + " and scale "
        + M(r"\sigma>0")
        + ".",
        M(r"\mathbb E[Y\mid x]=e^{\mu+\sigma^2/2}"),
        M(r"\operatorname{Var}(Y\mid x)=(e^{\sigma^2}-1)e^{2\mu+\sigma^2}"),
    ),
    "weibull": (
        M(r"f(y)=\frac{k}{\lambda}(y/\lambda)^{k-1}e^{-(y/\lambda)^k},\quad y>0"),
        "Scale " + M(r"\lambda>0") + "; shape " + M(r"k>0") + ".",
        M(r"\mathbb E[Y\mid x]=\lambda\Gamma(1+1/k)"),
        M(r"\operatorname{Var}(Y\mid x)=\lambda^2\{\Gamma(1+2/k)-\Gamma(1+1/k)^2\}"),
    ),
    "loglogistic": (
        M(r"F(y)=\{1+(y/\alpha)^{-\beta}\}^{-1},\quad y>0"),
        "Scale " + M(r"\alpha>0") + "; shape " + M(r"\beta>0") + ".",
        M(r"\mathbb E[Y\mid x]=\frac{\alpha\pi/\beta}{\sin(\pi/\beta)}")
        + " for "
        + M(r"\beta>1")
        + "; infinite otherwise.",
        M(
            r"\operatorname{Var}(Y\mid x)=\alpha^2\left\{\frac{2\pi/\beta}{\sin(2\pi/\beta)}-\left(\frac{\pi/\beta}{\sin(\pi/\beta)}\right)^2\right\}"
        )
        + " for "
        + M(r"\beta>2")
        + "; no finite variance otherwise.",
    ),
    "categorical": (
        M(r"\Pr(Y=j)=p_j,\quad j\in\{0,\ldots,C-1\}"),
        M(r"p_j\geq0,\quad\sum_{j=0}^{C-1}p_j=1"),
        "For the one-hot vector "
        + M(r"Z")
        + ", "
        + M(r"\mathbb E[Z\mid x]=p")
        + "; numeric label means depend on the encoding.",
        M(r"\operatorname{Cov}(Z\mid x)=\operatorname{diag}(p)-pp^{\mathsf T}")
        + " for one-hot "
        + M(r"Z")
        + ".",
    ),
    "ordinal": (
        M(
            r"\Pr(Y\leq j)=\operatorname{sigmoid}(c_j-\eta),\quad p_j=F(c_j-\eta)-F(c_{j-1}-\eta)"
        ),
        M(r"\eta\in\mathbb R,\quad-\infty=c_{-1}<c_0<\cdots<c_{C-2}<c_{C-1}=\infty"),
        "For integer-encoded labels, "
        + M(r"\mathbb E[Y\mid x]=\sum_{j=0}^{C-1}jp_j")
        + "; point prediction chooses a class.",
        M(
            r"\operatorname{Var}(Y\mid x)=\sum_{j=0}^{C-1}j^2p_j-\left(\sum_{j=0}^{C-1}jp_j\right)^2"
        )
        + " for that encoding.",
    ),
    "dirichlet": (
        M(
            r"f(y)=\frac{\Gamma(A)}{\prod_j\Gamma(\alpha_j)}\prod_jy_j^{\alpha_j-1},\quad A=\sum_j\alpha_j"
        )
        + "; "
        + M(r"y_j>0,\quad\sum_jy_j=1"),
        M(r"\alpha_j>0,\quad j=1,\ldots,D"),
        M(r"\mathbb E[Y_j\mid x]=\alpha_j/A"),
        M(r"\operatorname{Var}(Y_j\mid x)=\frac{\alpha_j(A-\alpha_j)}{A^2(A+1)}")
        + "; "
        + M(
            r"\operatorname{Cov}(Y_i,Y_j\mid x)=-\frac{\alpha_i\alpha_j}{A^2(A+1)},\quad i\ne j"
        ),
    ),
    "mvnormdiag": (
        M(r"Y_j\mid x\sim\mathcal N(\mu_j,\sigma_j^2)")
        + " conditionally independently; "
        + M(r"f(y\mid x)=\prod_{j=1}^Df_j(y_j\mid x)"),
        M(r"\mu_j\in\mathbb R,\quad\sigma_j>0,\quad j=1,\ldots,D"),
        M(r"\mathbb E[Y\mid x]=\boldsymbol\mu"),
        M(
            r"\operatorname{Cov}(Y\mid x)=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_D^2)"
        ),
    ),
    "quantile": (
        M(r"\rho_q(y-v)=(y-v)\{q-\mathbf1(y<v)\}")
        + "; sum over levels and average over observations. This is a loss, not a normalized density.",
        M(r"0<q_1<\cdots<q_Q<1")
        + "; predicted values are in configured quantile order.",
        "Not determined by a finite quantile vector; point output uses "
        + M(r"q=0.5")
        + " when present, otherwise the arithmetic average of the predicted quantiles.",
        "Not determined by the quantile outputs; ``predict_spread`` is unavailable.",
    ),
    "tweedie": (
        M(
            r"Y=\sum_{j=1}^NZ_j,\quad N\sim\operatorname{Poisson}\!\left(\frac{\mu^{2-p}}{\phi(2-p)}\right)"
        )
        + "; "
        + M(
            r"Z_j\sim\operatorname{Gamma}\!\left(\frac{2-p}{p-1},\ \mathrm{scale}=\phi(p-1)\mu^{p-1}\right)"
        ),
        M(r"\mu>0,\quad\phi>0,\quad1<p<2")
        + "; power is fixed or modeled according to the family contract.",
        M(r"\mathbb E[Y\mid x]=\mu"),
        M(r"\operatorname{Var}(Y\mid x)=\phi\mu^p"),
    ),
    "gumbls": (
        M(r"F(y)=\exp\{-e^{-(y-\mu)/\beta}\},\quad y\in\mathbb R"),
        M(r"\mu\in\mathbb R,\quad\beta>0"),
        M(r"\mathbb E[Y\mid x]=\mu+\gamma_E\beta")
        + ", where "
        + M(r"\gamma_E")
        + " is Euler's constant.",
        M(r"\operatorname{Var}(Y\mid x)=\pi^2\beta^2/6"),
    ),
    "gevlss": (
        M(r"F(y)=\exp\left\{-\left[1+\xi(y-\mu)/\sigma\right]^{-1/\xi}\right\}")
        + "; the "
        + M(r"\xi=0")
        + " limit is Gumbel.",
        M(r"\mu\in\mathbb R,\quad\sigma>0,\quad1+\xi(y-\mu)/\sigma>0")
        + "; "
        + M(r"\xi")
        + " also obeys the configured link bounds.",
        M(r"\mathbb E[Y\mid x]=\mu+\sigma\frac{\Gamma(1-\xi)-1}{\xi}")
        + " for "
        + M(r"\xi<1")
        + ", using its continuous limit at "
        + M(r"\xi=0")
        + "; undefined otherwise.",
        M(
            r"\operatorname{Var}(Y\mid x)=\sigma^2\frac{\Gamma(1-2\xi)-\Gamma(1-\xi)^2}{\xi^2}"
        )
        + " for "
        + M(r"\xi<1/2")
        + ", using its continuous limit at zero; no finite variance otherwise.",
    ),
    "shash": (
        M(
            r"Z=\sinh\!\left[\delta\operatorname{asinh}\!\left(\frac{Y-\mu}{\sigma\delta}\right)-\epsilon\right]\sim\mathcal N(0,1)"
        )
        + "; "
        + M(r"f_Y(y)=\varphi(z(y))|z'(y)|"),
        M(r"\mu,\epsilon\in\mathbb R,\quad\sigma,\delta>0"),
        M(
            r"\mathbb E[Y\mid x]=\mu+\sigma\delta\,\mathbb E\!\left[\sinh\!\left(\frac{\operatorname{asinh}Z+\epsilon}{\delta}\right)\right]"
        ),
        M(
            r"\operatorname{Var}(Y\mid x)=(\sigma\delta)^2\operatorname{Var}\!\left[\sinh\!\left(\frac{\operatorname{asinh}Z+\epsilon}{\delta}\right)\right]"
        )
        + "; finite for "
        + M(r"\delta>0")
        + ".",
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


def display_definition(name, definition):
    """Separate the main law from construction details for a display equation."""
    key = ALIASES.get(name, name)
    expressions = list(re.finditer(r":math:`([^`]+)`", definition))
    main = expressions[1] if key in {"normal", "studentt"} else expressions[0]
    if key in {"quasi", "quasipoisson", "quasibinomial"}:
        notes = "Quasi-likelihood specifies a mean-variance relationship, not a normalized probability density."
    elif key in {"cnorm", "clog", "cpois"}:
        notes = "The formula describes the latent law. Censoring contributes interval or one-sided probabilities."
    elif key == "mvnormdiag":
        notes = "Coordinates are conditionally independent; " + definition[
            main.end() :
        ].removeprefix(" conditionally independently; ")
    else:
        notes = ". ".join(
            part.strip(" ;.")
            for part in (definition[: main.start()], definition[main.end() :])
            if part.strip(" ;.")
        )
    return main.group(1), notes


def distribution_contract(name, backend):
    key = ALIASES.get(name, name)
    if key in {"zip", "zinb", "ziplss", "hurdlepoisson", "hurdlenegativebinom"}:
        hurdle = (
            key.startswith("hurdle")
            or key == "ziplss"
            or (backend == "statistical" and key == "zip")
        )
        nb = key in {"zinb", "hurdlenegativebinom"}
        base = "negative binomial" if nb else "Poisson"
        if hurdle:
            law = M(r"\Pr(Y=0)=\pi,\quad\Pr(Y=y>0)=(1-\pi)\frac{g(y)}{1-g(0)}")
            mean = M(r"\mathbb E[Y\mid x]=(1-\pi)m_+")
            variance = M(r"\operatorname{Var}(Y\mid x)=(1-\pi)v_++\pi(1-\pi)m_+^2")
            law += "; " + M(
                r"m_+=\frac{m}{1-g(0)},\quad v_+=\frac{v+m^2}{1-g(0)}-m_+^2"
            )
        else:
            law = M(r"\Pr(Y=0)=\pi+(1-\pi)g(0),\quad\Pr(Y=y>0)=(1-\pi)g(y)")
            mean = M(r"\mathbb E[Y\mid x]=(1-\pi)m")
            variance = M(r"\operatorname{Var}(Y\mid x)=(1-\pi)v+\pi(1-\pi)m^2")
        law += ". The base law " + M(r"g") + " is " + base + "; "
        law += M(r"m=\mu,\ v=\mu+\mu^2/r,\ r=1/\alpha") if nb else M(r"m=v=\lambda")
        domain = (
            M(r"0\leq\pi<1,\quad\mu>0,\quad\alpha>0")
            if nb
            else M(r"0\leq\pi<1,\quad\lambda>0")
        )
        return law, domain, mean, variance
    if key in {"quasi", "quasipoisson", "quasibinomial"}:
        variance = {
            "quasi": r"\phi V(\mu)",
            "quasipoisson": r"\phi\mu",
            "quasibinomial": r"\phi\mu(1-\mu)",
        }[key]
        domain = M(r"\mu\in\operatorname{dom}(g),\quad\phi>0")
        if key == "quasipoisson":
            domain = M(r"\mu>0,\quad\phi>0")
        if key == "quasibinomial":
            domain = M(r"0<\mu<1,\quad\phi>0")
        return (
            "Quasi-likelihood with "
            + M(r"\frac{\partial Q(y,\mu)}{\partial\mu}=\frac{y-\mu}{\phi V(\mu)}")
            + "; no normalized probability density.",
            domain,
            M(r"\mathbb E[Y\mid x]=\mu"),
            M(r"\operatorname{Var}(Y\mid x)=" + variance),
        )
    if key in {"cnorm", "clog", "cpois"}:
        law, domain, mean, variance = {
            "cnorm": (M(r"F(y)=\Phi((y-\mu)/\sigma)"), NORMAL[1], NORMAL[2], NORMAL[3]),
            "clog": (
                M(r"F(y)=\operatorname{sigmoid}((y-\mu)/s)"),
                M(r"\mu\in\mathbb R,\ s>0"),
                M(r"\mathbb E[Y\mid x]=\mu"),
                M(r"\operatorname{Var}(Y\mid x)=\pi^2s^2/3"),
            ),
            "cpois": POISSON,
        }[key]
        return (
            "Latent law: "
            + law
            + ". Censoring contributes interval or one-sided probabilities.",
            domain,
            mean + " for the latent response.",
            variance + " for the latent response.",
        )
    if key == "bcg":
        return (
            M(
                r"h_\lambda(y)=\begin{cases}(y^\lambda-1)/\lambda&\lambda\ne0,\\\log y&\lambda=0,\end{cases}\quad Z=h_\lambda(Y)"
            )
            + "; uncensored Gaussian transformation kernel "
            + M(
                r"\varphi\!\left(\frac{h_\lambda(y)-\mu}{\sigma}\right)\frac{y^{\lambda-1}}{\sigma}"
            ),
            M(r"Y>0,\ \mu,\lambda\in\mathbb R,\ \sigma>0")
            + "; inverse transformation requires "
            + M(r"1+\lambda Z>0")
            + ".",
            M(
                r"\mathbb E[Z\mid x]=\mu,\quad\mathbb E[Y\mid x]=\mathbb E[h_\lambda^{-1}(Z)\mid x]"
            )
            + "; at "
            + M(r"\lambda=0")
            + ", "
            + M(r"\mathbb E[Y\mid x]=e^{\mu+\sigma^2/2}")
            + ".",
            M(r"\operatorname{Var}(Z\mid x)=\sigma^2")
            + "; response variance is "
            + M(
                r"\mathbb E[h_\lambda^{-1}(Z)^2\mid x]-\mathbb E[h_\lambda^{-1}(Z)\mid x]^2"
            )
            + ". At "
            + M(r"\lambda=0")
            + ": "
            + M(r"(e^{\sigma^2}-1)e^{2\mu+\sigma^2}")
            + ". These formulas use unit prior weight.",
        )
    contract = DISTRIBUTIONS[key]
    if backend == "statistical" and name == "student_lss":
        return (
            STUDENT[0],
            M(r"\mu\in\mathbb R,\ \sigma>0,\ 2<\nu<\infty")
            + "; fixed "
            + M(r"\nu=5")
            + " by default.",
            NORMAL[2],
            M(r"\operatorname{Var}(Y\mid x)=\sigma^2\nu/(\nu-2)"),
        )
    if backend == "statistical" and name in {"beta_lss", "betar"}:
        return (
            M(r"Y\sim\operatorname{Beta}(\mu\phi,(1-\mu)\phi)"),
            M(r"0<\mu<1,\quad0<\phi<\infty,\quad0<Y<1"),
            M(r"\mathbb E[Y\mid x]=\mu"),
            M(r"\operatorname{Var}(Y\mid x)=\frac{\mu(1-\mu)}{1+\phi}"),
        )
    if (backend == "statistical" and name == "nb_lss") or (
        backend == "neural" and key == "negativebinom"
    ):
        return (
            NEGATIVE_BINOMIAL[0] + "; " + M(r"r=1/\alpha"),
            M(r"0<\mu<\infty,\quad0<\alpha<\infty"),
            NEGATIVE_BINOMIAL[2],
            M(r"\operatorname{Var}(Y\mid x)=\mu+\alpha\mu^2"),
        )
    if backend == "statistical" and name in {"gamma", "gammals"}:
        return (
            GAMMA[0] + "; " + M(r"a=1/\phi,\quad r=1/(\mu\phi)"),
            M(r"\mu>0,\quad\phi>0,\quad Y>0"),
            M(r"\mathbb E[Y\mid x]=\mu"),
            M(r"\operatorname{Var}(Y\mid x)=\phi\mu^2"),
        )
    return contract
