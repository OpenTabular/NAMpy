"""Mathematical criteria and optimization methods, deliberately kept separate."""

from inspect import cleandoc

from reference_theory import block

SELECTION = {
    "fixed": block(
        """Fixed smoothing solves the coefficient problem at the supplied
        penalty strengths. There is no data-driven outer choice of lambda.
        This is useful for scientifically chosen regularization or for
        comparing externally selected penalty settings.""",
        r"\widehat\beta_\lambda=\arg\min_\beta\{-\ell(\beta)+\tfrac12\beta^\mathsf TS_\lambda\beta\},\qquad S_\lambda=\sum_j\lambda_jS_j.",
        """Fixed lambda does not mean fixed coefficients, and it does not
        mean an unpenalized fit. Lambda=0 removes a particular penalty, while
        a positive value retains shrinkage. Penalty rescaling and weights
        determine the numerical meaning of a supplied lambda, so values
        cannot be transferred blindly between unrelated basis setups.""",
    ),
    "gcv": block(
        """For a Gaussian linear smoother, leave-one-out residuals involve
        individual leverages. GCV replaces those leverages by their average,
        producing a rotationally invariant approximation to prediction error.
        Supported non-Gaussian routes substitute model deviance for RSS.""",
        r"\operatorname{GCV}(\lambda)=\frac{D(\widehat\beta_\lambda)/n}{\{1-\gamma\operatorname{tr}(A_\lambda)/n\}^2}.",
        """D is residual sum of squares in the unweighted Gaussian case;
        tr(A) is effective degrees of freedom and gamma is the configured
        score inflation, normally one. The denominator penalizes flexibility.
        Small denominators can make scores unstable, and correlated errors
        undermine an ordinary leave-one-out interpretation. k still bounds
        the space over which GCV can choose a curve.""",
    ),
    "gacv": block(
        """Generalized approximate cross-validation uses the converged
        penalized IRLS fit to approximate held-out deviance. It includes a
        Pearson-type residual correction rather than simply squaring the
        GCV denominator.""",
        r"\operatorname{GACV}=\frac{D}{n}+\frac{2\gamma\,\operatorname{tr}(A)\,P}{n\{n-\gamma\operatorname{tr}(A)\}}.",
        """P is the weighted Pearson residual sum evaluated by the PIRLS
        kernel, D is its criterion deviance, and n is its observation count.
        Both the fitted weights and EDF depend on lambda. This is an
        approximation to a predictive criterion, not exact repeated refitting
        of every held-out observation. Availability depends on a supported
        PIRLS family and derivative route.""",
    ),
    "ubre": block(
        """UBRE estimates prediction risk when the response scale is known.
        It combines lack of fit with a linear effective-complexity correction;
        under the appropriate likelihood convention this is an AIC-type
        criterion up to scaling and constants.""",
        r"\operatorname{UBRE}=D/n-\phi+2\gamma\phi\operatorname{tr}(A)/n.",
        """Phi is the known scale, such as one for ordinary Poisson or
        binomial likelihoods. It is not an arbitrary fitted residual variance
        inserted after optimization. NAMpy rejects UBRE when its route has
        no known scale. Criterion values need consistent targets, weighting,
        scale and likelihood conventions to be comparable.""",
    ),
    "ml": block(
        """The random-effects interpretation treats penalized coefficient
        directions as Gaussian latent effects. Marginal maximum likelihood
        integrates those directions out and estimates their variance, leaving
        unpenalized coefficients as fixed effects.""",
        r"L_M(\lambda,\phi,\alpha)=\int p(y\mid\alpha,b,\phi)\,p(b\mid\lambda,\phi)\,db.",
        """The Gaussian integral is exact for compatible Gaussian designs;
        non-Gaussian models use a supported Laplace approximation. The
        integration automatically trades fit against the volume of plausible
        coefficient values. Unlike ordinary coefficient maximum likelihood,
        this selects smoothing strengths. Comparing models with different
        unpenalized fixed-effect spaces generally calls for ML rather than
        comparing their REML scores.""",
    ),
    "reml": block(
        """Restricted marginal likelihood additionally accounts for the
        estimation of unpenalized coefficient directions. In a Gaussian
        mixed-model view it can be expressed through residual contrasts that
        eliminate the fixed effects.""",
        r"y=X\alpha+Zb+\varepsilon,\quad C^\mathsf TX=0,\qquad L_R(\lambda,\phi)=p(C^\mathsf Ty\mid\lambda,\phi).",
        """Penalized and unpenalized directions come from the penalty's
        range and null spaces. Centering and rank handling therefore matter
        to the score as well as to coefficient uniqueness. For non-Gaussian
        models the supported restricted/Laplace construction plays the same
        role. REML values from different fixed-effect spaces do not represent
        the same residual contrasts and should not be compared as ordinary
        likelihoods. Wrappers default to REML where their route supports it.""",
    ),
    "laml": block(
        """Laplace approximate marginal likelihood replaces an intractable
        coefficient integral by a quadratic expansion around the penalized
        optimum. The local curvature contributes a log-determinant correction.""",
        r"\log L_{\mathrm{Lap}}\approx\ell(\widehat\beta)-\tfrac12\widehat\beta^\mathsf TS_\lambda\widehat\beta+\tfrac12\log|S_\lambda|_+-\tfrac12\log|H|+c.",
        """H is the penalized negative log-likelihood Hessian; the positive
        determinant uses only penalized directions. This schematic expression
        omits route-specific scale, fixed-effect and normalization terms.
        LAML names an approximation principle, not a new response family or
        a distinct guarantee of exact integration. NAMpy's laml spelling
        follows its supported REML/general-family dispatch; it is not an
        unrestricted integration option for every model.""",
    ),
    "p-ml": block(
        """The Pearson-scale ML variant changes how an unknown dispersion is
        supplied to a marginal-likelihood criterion. It uses Pearson residual
        information rather than claiming a new likelihood family.""",
        r"\widehat\phi_P=\frac{\sum_i w_i(y_i-\widehat\mu_i)^2/V(\widehat\mu_i)}{n-\mathrm{EDF}}.",
        """This displays the usual Pearson-scale interpretation; the
        implemented criterion retains its route's rank, observation and
        nuisance-parameter conventions. Known-scale families normalize p-ml
        to ML. Extended families normalize this spelling to REML in NAMpy,
        so the name alone does not establish a distinct fitted objective.""",
    ),
    "p-reml": block(
        """The Pearson-scale restricted-likelihood variant combines a
        Pearson estimate of unknown dispersion with restricted smoothing
        selection. Dispersion estimation and selection of lambda remain
        separate statistical roles.""",
        r"\widehat\phi_P\ \approx\ \frac{\text{Pearson residual sum}}{n-\mathrm{EDF}}.",
        """The denominator accounts for fitted effective complexity rather
        than the raw number of basis columns. Rank and nuisance conventions
        follow the selected route. Known-scale and extended families
        normalize p-reml to REML in NAMpy; this alias is not evidence that
        every family exposes a separate Pearson-profiled implementation.""",
    ),
    "ncv": block(
        """Neighborhood cross-validation removes specified observation
        sets, then evaluates prediction loss at associated held-out rows.
        Deletion sets and prediction sets need not be the same, which can
        represent local dependence or observation clusters.""",
        r"\operatorname{NCV}(\lambda)=\sum_k\sum_{i\in P_k}d\{y_i,\widehat\mu_i^{(-D_k)}(\lambda)\}.",
        """D_k is a deletion neighborhood and P_k its prediction rows.
        NAMpy approximates the coefficient change using the fitted
        likelihood's derivative/curvature information instead of fully
        refitting every deletion. A neighborhood should reflect the intended
        prediction problem; ordinary leave-one-row-out can be optimistic
        when nearby rows share information. The nei contract specifies
        indexing and membership; ncv_threads does not enable parallel kernels.""",
    ),
    "qncv": block(
        """Quadratic NCV retains the same neighborhood deletion structure
        but approximates the held-out loss itself by a second-order expansion
        around the full-data fit.""",
        r"d(\widehat\eta+\Delta\eta)\approx d(\widehat\eta)+\nabla d^\mathsf T\Delta\eta+\tfrac12\Delta\eta^\mathsf T\nabla^2d\,\Delta\eta.",
        """Thus both coefficient deletion and loss evaluation use local
        information. Accuracy depends on the size of the deletion-induced
        change and the quality of the local curvature approximation. QNCV
        does not make neighborhoods independent or replace the need for
        genuinely held-out validation. Compare it with NCV using the same
        neighborhood definition and response-loss convention.""",
    ),
    "outer_newton": block(
        """Outer Newton optimizes a selected smoothing criterion in log
        penalty coordinates. Each criterion evaluation solves the coefficient
        problem at the current smoothing parameters; derivatives account for
        the resulting change in fitted coefficients.""",
        r"\rho_j=\log\lambda_j,\qquad H_C(\rho)\Delta\rho=-\nabla C(\rho).",
        """Log coordinates enforce positive free penalties. Newton uses
        criterion curvature to scale updates, with stabilization and step
        reduction when needed. An inner fit that has not converged can give
        unreliable outer derivatives. Availability requires the family and
        design to provide the derivative orders needed by the selected
        criterion; choosing Newton does not supply missing derivatives.""",
    ),
    "bfgs": block(
        """BFGS is a quasi-Newton method: it learns a local inverse-Hessian
        approximation from changes in criterion gradients. It optimizes the
        same criterion as an exact outer Newton route when that combination
        is supported.""",
        r"s_k=\rho_{k+1}-\rho_k,\quad y_k=\nabla C_{k+1}-\nabla C_k,\qquad M_{k+1}y_k=s_k.",
        """M approximates inverse criterion curvature; a line search checks
        candidate progress. This avoids an explicit exact outer Hessian but
        still requires trustworthy criterion/gradient evaluations. It does
        not change REML into a different statistical objective, and finite
        coefficients alone do not establish outer convergence.""",
    ),
    "efs": block(
        """Extended Fellner-Schall updates use penalty quadratic forms and
        curvature traces to update variance-component/smoothing coordinates.
        They avoid the full high-order outer derivative machinery required
        by exact Newton smoothing selection.""",
        r"\lambda_j^{\mathrm{new}}\ \propto\ \frac{\text{effective dimension for penalty }j}{\widehat\beta^\mathsf TS_j\widehat\beta}.",
        """This is the variance-component balance underlying the update,
        not a literal replacement for the implementation's scaled trace,
        safeguards and bounds. Repeated updates alternate with coefficient
        fitting. EFS is especially useful for supported general families
        whose higher derivatives are unavailable, including twlss. It is
        an optimization/update method, not a cross-validation criterion.""",
    ),
    "optim": block(
        """The optim route selects a numerical optimizer for the supported
        smoothing objective. Method settings control how candidate penalty
        values are proposed, while the family/criterion dispatch determines
        what quantity each candidate is scored against.""",
        r"\min_{\rho\in\mathbb R^s}C\{\widehat\beta(\exp\rho),\exp\rho\}.",
        """Gradient-based and derivative-free methods have different costs
        and stopping rules. Neither turns an unsupported family/criterion
        pairing into a supported one. Record optim_method, the nested optim
        settings and convergence information when comparing runs; the word
        optim alone does not identify the algorithm used.""",
    ),
    "nlm": block(
        """The NLM-style route minimizes a smooth scalar objective using
        local numerical derivative and curvature information. In NAMpy this
        dispatch is restricted to supported SCAM transformed-coefficient
        fits rather than ordinary unrestricted GAM fitting.""",
        r"C(\rho+\delta)\approx C(\rho)+g^\mathsf T\delta+\tfrac12\delta^\mathsf TH\delta.",
        """Finite-difference steps, parameter scales, tolerances and step
        limits influence derivative quality and convergence. The transformed
        coefficient solve must remain valid at perturbed smoothing values.
        This describes the local numerical model, not a guarantee that the
        Hessian is exact or that every GAM objective accepts this route.""",
    ),
}


def selection_theory(name):
    return (
        "Theory and interpretation\n-------------------------\n\n"
        + SELECTION[name]
        + cleandoc("""See :doc:`/statistical_reference/selection/theory` for the full
        coefficient/selection loop, notation and comparison rules. The
        configuration and support contracts below describe NAMpy's routes.""")
        + "\n\n"
    )
