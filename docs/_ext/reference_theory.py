"""Authored mathematical explanations for registered reference features.

These explanations complement live signatures. Missing coverage is an error:
adding a feature requires an explicit decision about its mathematical meaning.
"""

from inspect import cleandoc


def block(prose, equation, interpretation):
    return (
        cleandoc(prose)
        + "\n\n.. math::\n\n"
        + "\n".join("   " + line for line in cleandoc(equation).splitlines())
        + "\n\n"
        + cleandoc(interpretation)
        + "\n\n"
    )


SMOOTHS = {
    "tp": block(
        """A thin-plate smooth starts with a rotation-invariant measure of roughness.
        In d dimensions, derivatives of total order m enter the penalty below,
        with 2m>d. Its zero-penalty functions are polynomials of degree less than m.
        A truncated eigendecomposition of the radial-kernel part supplies a
        low-rank regression basis; polynomial columns complete the null space.""",
        r"J_m(f)=\int_{\mathbb R^d}\sum_{|\alpha|=m}\frac{m!}{\alpha!}\{D^\alpha f(x)\}^2\,dx.",
        """For d=1 and m=2, J is integrated squared curvature and its null space
        contains straight lines. Increasing lambda favors that null space;
        it does not generally delete the term. Increasing k retains more kernel
        directions, while m changes the definition of roughness. Isotropy makes
        sense when joint coordinate distances have a meaningful scale; use a
        tensor product for variables such as time and temperature with unrelated
        units. Training locations or the selected construction subset define
        a fixed basis that is reused at prediction time.""",
    ),
    "cr": block(
        """A cubic regression spline joins cubic polynomial pieces at knots,
        matching their first two derivatives. A natural boundary condition
        sets curvature to zero at the outer knots. Expressing the curve in a
        finite knot basis turns the continuous roughness functional into a
        quadratic coefficient penalty.""",
        r"f(x)=\sum_r b_r(x)\beta_r,\qquad J(f)=\int\{f''(x)\}^2dx=\beta^\mathsf TS\beta.",
        """Constant and linear functions have zero curvature. After ordinary
        centering removes the constant/intercept ambiguity, strong smoothing
        can leave a linear trend. k controls the available knot representation,
        while lambda controls its curvature. Natural cubic tails extrapolate
        linearly, so uncertainty and scientific plausibility outside the
        observed range still require care.""",
    ),
    "ps": block(
        """A P-spline represents f with locally supported B-splines and penalizes
        differences between neighboring coefficients. Local support means a
        basis function is nonzero over only a small knot interval. The spline
        degree and difference order are separate choices; the familiar
        second-difference penalty discourages abrupt coefficient slopes.""",
        r"J(\beta)=\|D_q\beta\|^2,\qquad S=D_q^\mathsf TD_q,\qquad (D_2\beta)_r=\beta_{r+2}-2\beta_{r+1}+\beta_r.",
        """This is a discrete coefficient penalty rather than an exact integral
        of function derivatives. Its null space contains coefficient sequences
        with zero q-th differences. Enough knots supply representational
        capacity; smoothing then suppresses unsupported variation. A larger
        number of coefficients alone is not evidence of a more complex fitted
        curve. Knot spacing and boundary behavior remain part of the model.""",
    ),
    "bs": block(
        """Derivative-penalized B-splines keep the local B-spline representation
        but measure roughness in function space. Differentiating each basis
        function and integrating products constructs the penalty exactly up
        to the numerical integration used by the constructor.""",
        r"S^{(q)}_{rs}=\int b_r^{(q)}(x)b_s^{(q)}(x)dx,\qquad J_q(f)=\int\{f^{(q)}(x)\}^2dx.",
        """With multiple requested derivative orders, each penalty gets its own
        smoothing coordinate. A q-th derivative penalty leaves polynomials of
        degree below q unpenalized when those polynomials are in the basis.
        Unlike a difference penalty, the derivative construction explicitly
        incorporates the covariate metric and knot spacing. Basis order must
        be high enough to define the requested derivatives.""",
    ),
    "ds": block(
        """Duchon splines generalize thin-plate roughness by weighting derivative
        energy in the Fourier domain. In the schematic expression below,
        omega is frequency, m controls derivatives, and s adjusts the spectral
        weighting. Only admissible combinations give a well-defined spline.""",
        r"J_{m,s}(f)\ \propto\ \int\|\omega\|^{2s}\sum_{|\alpha|=m}\frac{m!}{\alpha!}\left|\widehat{D^\alpha f}(\omega)\right|^2d\omega.",
        """Changing the spectral weight can allow a lower derivative order and
        smaller polynomial null space than a conventional thin-plate spline
        in the same input dimension. The reduced-rank representation still
        separates penalized kernel directions from unpenalized polynomial
        directions. This is an isotropic multivariate smoother: relative
        coordinate scaling affects the notion of distance. The m contract
        below specifies the constructor's accepted order/weight pair.""",
    ),
    "gp": block(
        """A Gaussian-process smooth starts from a covariance kernel K(x,x').
        Correlation decreases with separation according to the chosen kernel
        and range. A reduced-rank covariance eigensystem yields basis columns
        and a precision penalty, with any explicit trend/null-space directions
        handled separately.""",
        r"f\sim\mathcal{GP}(0,\tau^2K),\qquad J(f)\approx\beta^\mathsf TS_K\beta.",
        """The kernel selects which patterns are plausible; its range selects
        a distance scale; lambda controls their amplitude relative to fitting
        error. These are different roles. NAMpy fits the resulting finite
        penalized design rather than performing unrestricted dense GP
        hyperparameter inference. The selected kernel, range and construction
        locations are retained for prediction. Coordinate units should match
        the covariance-distance interpretation.""",
    ),
    "sos": block(
        """A spherical spline uses angular separation on the sphere rather than
        Euclidean distance between latitude/longitude numbers. The kernel
        depends on the dot product of unit position vectors; its order selects
        the spherical roughness spectrum and associated null space.""",
        r"u(\varphi,\lambda)=(\cos\varphi\cos\lambda,\cos\varphi\sin\lambda,\sin\varphi),\qquad \cos\theta=u^\mathsf Tu'.",
        """This respects the longitude seam and shrinking longitude distances
        near the poles. Inputs are latitude then longitude in degrees; the
        displayed formula uses radians internally. Smoothness is isotropic on
        the sphere, not on a rectangular map projection. The m branches have
        different kernel/null-space definitions, so a planar cubic-curvature
        interpretation does not transfer unchanged.""",
    ),
    "ad": block(
        """An adaptive spline allows the strength of smoothing to vary over
        the covariate domain. A low-dimensional basis represents the local
        penalty weights, producing several positive-semidefinite penalty
        matrices rather than one global roughness matrix.""",
        r"S(\lambda)=\sum_h\lambda_hS_h,\qquad J(\beta)=\sum_h\lambda_h\beta^\mathsf TS_h\beta.",
        """Different penalty components emphasize different regions, allowing
        a curve with a sharp feature in one region and a smooth trend elsewhere.
        The function basis dimension and the penalty-weight basis dimension
        control different kinds of flexibility. Many local smoothing weights
        require enough data to identify them; adaptive smoothing is more
        demanding than fitting one ordinary P-spline. The two-dimensional
        construction extends this idea over marginal directions.""",
    ),
    "so": block(
        """A soap-film smooth separates a bounded-domain surface into a boundary
        film and an interior component that vanishes on the boundary. The film
        extends boundary values harmonically into the domain. Interior basis
        functions are constructed by solving a boundary-value problem.""",
        r"f=f_{\partial\Omega}+f_0,\qquad f_0|_{\partial\Omega}=0,\qquad J_0(f_0)=\int_\Omega\{\Delta f_0\}^2dx.",
        """Boundary and interior variation have separate penalties. The domain
        geometry prevents information from crossing excluded land or other
        barriers as an unconstrained planar smoother would. Interior knots
        set the interior approximation space; boundary sampling and the
        numerical grid control the boundary-value approximation. Predictions
        are meaningful inside the specified domain, subject to its membership
        rules, rather than across arbitrary exterior locations.""",
    ),
    "re": block(
        """A random-effect smooth constructs a parametric design Z from its
        supplied variables, then shrinks its coefficients with a precision
        penalty. A single categorical variable gives group indicators;
        additional variables form the corresponding parametric product design.""",
        r"\eta=X\alpha+Zb,\qquad b\sim N(0,\tau^2I),\qquad J(b)=b^\mathsf Tb.",
        """For a Gaussian response with variance phi, the usual penalty
        coordinate satisfies lambda=phi/tau^2 before internal rescaling.
        Large lambda pools group deviations strongly toward zero; small
        lambda allows more between-group variation. A supplied structured
        precision replaces I where supported. This is partial pooling of
        coefficients, not smoothing over an ordered group-number axis.
        Prediction for unseen groups follows the documented runtime policy.""",
    ),
    "mrf": block(
        """A Markov random field treats discrete regions as vertices in a graph.
        For an undirected adjacency matrix A and degree matrix D, the graph
        Laplacian penalizes differences between neighboring region effects.""",
        r"Q=D-A,\qquad J(b)=b^\mathsf TQb=\sum_{\{r,s\}\in E}(b_r-b_s)^2.",
        """Connected regions borrow information; non-neighbors interact only
        through graph paths. Each connected component contributes a constant
        null-space direction, so centering and rank handling matter for
        disconnected graphs. A reduced-rank constructor retains a subset of
        graph directions; a supplied precision may define a different
        quadratic relationship. The model is a discrete regional surface,
        not a continuous interpolation between polygon centroids.""",
    ),
    "fs": block(
        """A factor-smooth interaction duplicates a numerical basis for every
        factor level. Each observation activates only its own level's block.
        Shared smoothing coordinates pool the degree of wiggliness across
        levels, while null-space penalties also shrink level-specific offsets
        and low-order trends.""",
        r"f(x,g)=\sum_{l=1}^{L}I(g=l)B(x)\beta_l,\qquad J=\sum_l\beta_l^\mathsf TS(\lambda)\beta_l.",
        """Thus a sparsely observed level can shrink toward zero throughout
        its curve, rather than retaining an arbitrary unpenalized intercept
        or slope. This construction is different from explicitly centering
        deviations around one common smooth. If a common effect and
        sum-to-zero deviations are the scientific target, consider sz.
        Factor levels and their basis state must be consistent at prediction.""",
    ),
    "sz": block(
        """A sum-to-zero factor smooth represents level-specific deviations
        from a common smooth. Constraints remove the average coefficient
        direction across factor levels, making a separate shared main smooth
        interpretable.""",
        r"\eta=f_0(x)+f_g(x),\qquad \sum_{l=1}^{L}\beta_{lr}=0\quad\text{for each basis coefficient }r.",
        """Because the same basis coordinates are constrained across levels,
        the deviations sum to zero as functions, not merely at one location.
        With multiple factors, the construction excludes the appropriate
        lower-order factor effects under its documented constraints. A curve
        for a level is reconstructed by adding its deviation to the common
        smooth and other model terms. This is an identifiability construction,
        not evidence that each deviation is well estimated.""",
    ),
    "te": block(
        """A full tensor product combines marginal bases without requiring
        the variables to share units. Every product of marginal basis columns
        enters the candidate joint surface. A separate penalty measures
        roughness along each margin while holding the other coordinate fixed.""",
        r"f(x,z)=\sum_{r=1}^{k_x}\sum_{s=1}^{k_z}\beta_{rs}b_r(x)c_s(z),\qquad S_x=S^{(x)}\otimes I,\quad S_z=I\otimes S^{(z)}.",
        """The raw basis width is k_x*k_z, reduced by absorbed constraints;
        higher-dimensional products follow the same rule. Separate lambdas
        permit different smoothness along x and z. A full surface includes
        directions resembling marginal main effects, so adding separate main
        smooths can create overlapping representations. Use ti when the goal
        is a distinct interaction alongside explicitly modeled main effects.
        Changing a marginal basis changes both approximation and penalty.""",
    ),
    "ti": block(
        """A tensor interaction first applies the requested marginal
        identifiability constraints and then forms their product. Centering
        marginal bases removes the constant directions that would otherwise
        reproduce lower-order effects.""",
        r"\eta=f_x(x)+f_z(z)+f_{xz}(x,z),\qquad B_{xz}=\widetilde B_x\odot\widetilde B_z.",
        """Here the row-wise tensor product takes all column products for
        each observation. With centered margins, the joint term describes
        departures from the additive main effects. For k_x and k_z raw
        marginal widths, the interaction often has (k_x-1)(k_z-1) columns,
        but the actual constraints and null spaces determine its rank.
        mc can disable selected marginal constraints; doing so changes the
        decomposition and can reintroduce overlap. Interpretation requires
        inspecting all main and interaction terms together.""",
    ),
    "t2": block(
        """The t2 construction diagonalizes each marginal penalty and splits
        its basis into unpenalized null-space and penalized range-space
        columns. Tensor products of these pieces form distinct functional
        blocks with non-overlapping penalty supports.""",
        r"B_x=[N_x,R_x],\quad B_z=[N_z,R_z],\qquad B_{xz}=[N_x\odot N_z,\ N_x\odot R_z,\ R_x\odot N_z,\ R_x\odot R_z].",
        """Blocks with range-space factors receive penalties; the all-null
        block remains unpenalized unless additional selection changes it.
        full controls whether null-space components are grouped or kept
        separate. ord retains blocks according to their number of range-space
        margins, rather than selecting polynomial degrees. Consequently t2
        and te can span similar surfaces but regularize them differently.
        Additional blocks also mean additional smoothing parameters.""",
    ),
}


def smooth_theory(name):
    from smooth_contracts import SHAPE_NAMES

    if name in {"ts", "cs"}:
        parent = "tp" if name == "ts" else "cr"
        text = SMOOTHS[parent] + block(
            """The shrinkage variant changes the zero eigenvalues of the ordinary
            roughness penalty into small positive values.""",
            r"S=U\operatorname{diag}(d)U^\mathsf T\quad\longrightarrow\quad S_* = U\operatorname{diag}(d_*)U^\mathsf T,\qquad d_{*r}>0.",
            """Heavy smoothing can now suppress every remaining coefficient
            direction, including the old polynomial null space. This permits
            effective term removal. It differs from a separate null-space
            selection penalty, which estimates an additional lambda instead
            of fixing the relative shrinkage strength.""",
        )
    elif name in {"cc", "cp"}:
        text = block(
            """The cyclic construction identifies the ends of the domain.
            For cyclic cubics, function value and the first two derivatives
            match at the boundary; cyclic P-splines wrap the basis and
            coefficient differences.""",
            r"f(a)=f(b),\qquad f^{(j)}(a)=f^{(j)}(b)\quad\text{for the matched derivative orders}.",
            """Periodicity removes a discontinuity at the seam, as needed for
            hour of day or season. It replaces the ordinary open-boundary
            behavior described above; a nonconstant straight line is not a
            periodic null-space function. Supply boundaries that represent a
            full scientific period, rather than relying on an accidentally
            truncated observed range.""",
        )
        text += (
            block(
                """The cyclic cubic penalty still measures integrated
                squared curvature, now in a periodic rather than natural
                open-boundary function space.""",
                r"J(f)=\int_a^b\{f''(x)\}^2dx=\beta^\mathsf TS_{\mathrm{cyclic}}\beta.",
                """Its smoothest periodic direction is constant. After
                centering against an intercept, increasing lambda can suppress
                the remaining seasonal variation. k controls the periodic
                approximation space; lambda controls how much variation is
                supported within that space.""",
            )
            if name == "cc"
            else block(
                """The cyclic P-spline wraps both its B-spline support and
                coefficient differences across the seam. Coefficients at
                the final knot positions connect back to the first.""",
                r"J(\beta)=\|D_{q,\mathrm{cyclic}}\beta\|^2.",
                """A wraparound difference penalty removes the artificial
                endpoint discontinuity of an open P-spline. It penalizes
                coefficient differences rather than an exact derivative
                integral, so spline order and penalty order have distinct
                roles. The stored period and fitted knot setup must also
                be used for new rows.""",
            )
        )
    elif name in {"sf", "sw"}:
        text = (
            SMOOTHS["so"]
            + cleandoc(
                """This constructor retains only the boundary film and its penalty;
            it does not independently model interior departures.\n\n"""
                if name == "sf"
                else """This constructor retains only the zero-boundary interior
            component. A nonzero boundary effect requires a separate film.\n\n"""
            )
            + "\n\n"
        )
    elif name in {"sc", "scad"}:
        text = (
            SMOOTHS["ps" if name == "sc" else "ad"]
            + cleandoc(
                """The sc/scad constructor names identify B-spline construction
            branches. They do not impose a sign constraint on coefficients or
            derivatives. Use an explicit SCOP basis such as mpi for monotonicity.\n\n"""
            )
            + "\n\n"
        )
    elif name in SHAPE_NAMES:
        text = shape_theory(name)
    else:
        text = SMOOTHS[name]
    return (
        "Theory and interpretation\n-------------------------\n\n"
        + text
        + cleandoc("""For coefficient coordinates, centering, null spaces and EDF, see
        :doc:`/statistical_reference/smooths/theory`. The formula and runtime
        contracts below specify which mathematical options NAMpy exposes.""")
        + "\n\n"
    )


UNIVARIATE_SHAPES = {
    "mpi": (r"f'(x)\geq0", "nonnegative first differences imply an increasing curve"),
    "mpd": (
        r"f'(x)\leq0",
        "reversed signed first differences imply a decreasing curve",
    ),
    "micx": (
        r"f'(x)\geq0,\ f''(x)\geq0",
        "positive increments and increasing slopes give increase and convexity",
    ),
    "micv": (
        r"f'(x)\geq0,\ f''(x)\leq0",
        "positive increments and decreasing slopes give increase and concavity",
    ),
    "mdcx": (
        r"f'(x)\leq0,\ f''(x)\geq0",
        "negative slopes that increase toward zero give decrease and convexity",
    ),
    "mdcv": (
        r"f'(x)\leq0,\ f''(x)\leq0",
        "negative slopes that become more negative give decrease and concavity",
    ),
    "cx": (
        r"f''(x)\geq0",
        "nonnegative curvature differences allow either sign of the slope",
    ),
    "cv": (
        r"f''(x)\leq0",
        "nonpositive curvature differences allow either sign of the slope",
    ),
    "po": (r"f(x)\geq0", "positive basis coefficients constrain the function itself"),
    "ipo": (
        r"f(x)\geq0,\ f'(x)\geq0",
        "positive cumulative coefficients give a positive increasing function",
    ),
    "dpo": (
        r"f(x)\geq0,\ f'(x)\leq0",
        "a reversed cumulative construction gives a positive decreasing function",
    ),
    "cpop": (
        r"f(x)\geq0,\ f(a)=f(b)",
        "a cyclic positive basis joins a nonnegative function across its seam",
    ),
    "miso": (
        r"f'(x)\geq0,\ f(a)=0",
        "the increasing curve is anchored to zero at its starting endpoint",
    ),
    "mifo": (
        r"f'(x)\geq0,\ f(b)=0",
        "the increasing curve is anchored to zero at its ending endpoint",
    ),
    "lmpi": (
        r"f'(x)\geq0\ \text{for }x\leq x_c",
        "increase is required before xc, while the later coefficient block is unconstrained",
    ),
    "lipl": (
        r"f'(x)\geq0\ \text{for }x<x_c,\quad f'(x)=0\ \text{for }x>x_c",
        "increase transitions to a plateau at the supplied change point xc",
    ),
}
BIVARIATE_SHAPES = {
    "tedmi": r"f_x\geq0,\ f_z\geq0",
    "tedmd": r"f_x\leq0,\ f_z\leq0",
    "temicx": r"f_x\geq0,\ f_{zz}\geq0",
    "temicv": r"f_x\geq0,\ f_{zz}\leq0",
    "tedecx": r"f_x\leq0,\ f_{zz}\geq0",
    "tedecv": r"f_x\leq0,\ f_{zz}\leq0",
    "tecxcx": r"f_{xx}\geq0,\ f_{zz}\geq0",
    "tecvcv": r"f_{xx}\leq0,\ f_{zz}\leq0",
    "tecxcv": r"f_{xx}\geq0,\ f_{zz}\leq0",
    "tescx": r"f_{zz}\geq0",
    "tescv": r"f_{zz}\leq0",
    "tesmi1": r"f_x\geq0",
    "tesmd1": r"f_x\leq0",
    "tesmi2": r"f_z\geq0",
    "tesmd2": r"f_z\leq0",
    "tismi": r"f_x\geq0",
    "tismd": r"f_x\leq0",
}


def shape_theory(name):
    base = name.removesuffix("by")
    if base in UNIVARIATE_SHAPES:
        inequality, description = UNIVARIATE_SHAPES[base]
        text = block(
            """Shape-constrained P-splines build cumulative/difference maps
            into a B-spline representation. Selected coefficient directions
            are mapped through a positive transform; remaining directions
            stay unconstrained. In this construction, """
            + description
            + ".",
            inequality + r",\qquad f(x)=B(x)\Sigma c(\theta).",
            """The displayed condition is enforced structurally over the spline
            domain, rather than checked only at the observed rows. Smoothing
            still controls variation within the allowed shapes. The fitted
            predictor is nonlinear in the free parameters theta, so ordinary
            unconstrained least-squares coefficient updates do not apply.
            A shape constraint is a modeling assumption, not a conclusion
            from a significance test. Derivative statements concern this term
            on the additive predictor scale, holding other terms fixed.""",
        )
        if base in {"lmpi", "lipl"}:
            text += "The change point xc is supplied, not estimated by smoothing selection. Moving it changes the constraint and basis setup.\n\n"
        if base == "mifo":
            text += "An increasing function ending at zero is nonpositive before that endpoint; this is not a positivity constraint.\n\n"
        if name.endswith("by"):
            text += "For a numeric multiplier v the model contribution is v*f(x). A negative v reverses the direction of its derivative with respect to x; the constraint is on the underlying smooth, not every multiplied contribution.\n\n"
    else:
        inequality = BIVARIATE_SHAPES[name]
        text = block(
            """A bivariate shape spline forms tensor products of marginal
            B-splines, then applies cumulative coefficient maps and positivity
            transforms to enforce the indicated partial-derivative signs.
            Subscripts x and z denote the first and second input covariates.""",
            inequality + r",\qquad f(x,z)=\{B_x(x)\otimes B_z(z)\}\Sigma c(\theta).",
            """Two marginal roughness penalties regulate the surface along
            its coordinates while retaining the shape restrictions. Where
            only one marginal derivative is shown, the other is unrestricted.
            Separate convexity in x and z is not a joint positive-semidefinite
            Hessian condition: a mixed derivative can prevent joint convexity.
            Constraints apply on the link/term scale and are not causal
            statements about how changing an input affects an outcome.""",
        )
        if name in {"tismi", "tismd"}:
            text += "This is the interaction parameterization with its own identifiability construction. It does not assert a sign for the mixed partial derivative; inspect the accompanying marginal effects when interpreting a full model.\n\n"
    text += "The coefficient transformation and penalty coordinates must be evaluated together; the shape fitting route retains their derivatives. See :doc:`/user_guide/shape_constrained_gams` for centering, inference and fitting restrictions.\n\n"
    return text
