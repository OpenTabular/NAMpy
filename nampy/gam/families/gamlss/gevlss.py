from __future__ import annotations

from typing import Any

import numpy as np

from ..._mgcv_constants import FAMILY_EPS
from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from .._function_maps import LogLink
from ._base import (
    GamlssFamily,
    _AdaptedLinkInfo,
    _IdentityLinkInfo,
    _pen_reg,
    _qr_coef_pivoted,
)


class _ShiftedLogitLinkInfo:
    """
    Shifted logit link confining xi to (-1, 0.5):
        linkinv(eta) = 1.5 * sigmoid(eta) - 1
        linkfun(xi)  = logit((xi + 1) / 1.5)
        mu_eta(eta)  = 1.5 * sigmoid(eta) * (1 - sigmoid(eta))

    Mirrors mgcv gevlss stats[[3]] with link="logit".
    """

    name = "shifted_logit"

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return 1.5 * self._sigmoid(np.asarray(eta, dtype=np.float64)) - 1.0

    def linkfun(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        p = (xi + 1.0) / 1.5
        return np.log(p / (1.0 - p))

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        s = self._sigmoid(np.asarray(eta, dtype=np.float64))
        return 1.5 * s * (1.0 - s)

    def d2link(self, xi: np.ndarray) -> np.ndarray:
        """d^2 eta / d xi^2.  Mirrors mgcv d2link for shifted logit."""
        xi = np.asarray(xi, dtype=np.float64)
        mu = (xi + 1.0) / 1.5
        return (1.0 / (1.0 - mu) ** 2 - 1.0 / mu**2) / 1.5**2

    def d3link(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        mu = (xi + 1.0) / 1.5
        return (2.0 / (1.0 - mu) ** 3 + 2.0 / mu**3) / 1.5**3

    def d4link(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        mu = (xi + 1.0) / 1.5
        return (6.0 / (1.0 - mu) ** 4 - 6.0 / mu**4) / 1.5**4


# ---------------------------------------------------------------------------
# gevlss: GEV location-scale-shape  (mgcv: gamlss.r::gevlss)
# ---------------------------------------------------------------------------


class GevlssFamily(GamlssFamily):
    """
    GEV location-scale-shape family with three linear predictors:
      1. location μ (identity or log link)
      2. log scale ρ = log(σ) (identity link)
      3. shape ξ ∈ (-1, 0.5) (shifted-logit link)

    GEV distribution: F(y) = exp(-(1 + ξ*(y-μ)/σ)^(-1/ξ))

    Mirrors mgcv ``gevlss(link=list("identity","identity","logit"))``.

    Note: family now exposes analytic outer gradient and Hessian terms through
    ``gam.fit5``.
    """

    name = "gevlss"
    family_class = "general"
    nlp = 3
    n_linear_predictors = 3

    supports_laml = True
    supports_ml = True
    supports_reml = True
    supports_gcv = False
    supports_ubre = False
    supports_pirls = False
    supports_analytic_outer_derivatives = False
    supports_closed_form_solve = False
    supports_exact_pirls_first_derivatives = False
    supports_exact_pirls_second_derivatives = False
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, link=("identity", "identity", "logit")):
        super().__init__()

        ok1 = ("identity", "log")
        ok2 = ("identity",)
        ok3 = ("identity", "logit")

        link1_name = link[0] if isinstance(link[0], str) else "identity"
        link2_name = link[1] if isinstance(link[1], str) else "identity"
        link3_name = link[2] if isinstance(link[2], str) else "logit"

        if link1_name not in ok1:
            raise ValueError(f"Link {link1_name!r} not available for mu of gevlss.")
        if link2_name not in ok2:
            raise ValueError(
                f"Link {link2_name!r} not available for log-sigma of gevlss."
            )
        if link3_name not in ok3:
            raise ValueError(f"Link {link3_name!r} not available for xi of gevlss.")

        if link1_name == "identity":
            linfo1 = _IdentityLinkInfo()
        else:
            linfo1 = _AdaptedLinkInfo(LogLink(eps=FAMILY_EPS), link1_name)

        linfo2 = _IdentityLinkInfo()

        if link3_name == "logit":
            linfo3 = _ShiftedLogitLinkInfo()
        else:
            linfo3 = _IdentityLinkInfo()

        self.linfo = [linfo1, linfo2, linfo3]
        self.tri = trind_generator(3)
        self.link_names = (link1_name, link2_name, link3_name)
        self.link_name = f"({link1_name}, {link2_name}, {link3_name})"

    def ll(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        coef: np.ndarray,
        weights: np.ndarray,
        offset: Any = None,
        deriv: int = 0,
        d1b: Any = 0,
        d2b: Any = 0,
        fh: Any = None,
        D: Any = None,
        **kw,
    ) -> dict[str, Any]:
        """
        Log-likelihood and derivatives for the GEV model.

        Mirrors mgcv ``gevlss$ll`` (auto-generated derivative expressions).
        Third/fourth derivative orders (deriv>1) return zeros.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        n = len(y)
        sandwich = bool(kw.get("sandwich", False))

        eta_mat = self._eta_matrix_from_inputs(
            X,
            jj,
            coef,
            offset=offset,
            eta=kw.get("eta", None),
        )
        eta = np.asarray(eta_mat[:, 0], dtype=np.float64)
        etar = np.asarray(eta_mat[:, 1], dtype=np.float64)
        etax = np.asarray(eta_mat[:, 2], dtype=np.float64)

        mu = self.linfo[0].linkinv(eta)  # location
        rho = self.linfo[1].linkinv(etar)  # log scale
        xi = self.linfo[2].linkinv(etax)  # shape

        # Clamp xi away from zero (avoids xi=0 branch)
        eps_xi = 1e-7
        xi = np.where((xi >= 0.0) & (xi < eps_xi), eps_xi, xi)
        xi = np.where((xi < 0.0) & (xi > -eps_xi), -eps_xi, xi)

        ymu = y - mu
        sigma_inv = np.exp(-rho)  # = bb1 = 1/sigma
        aa0 = xi * ymu * sigma_inv  # = xi*(y-mu)/exp(rho)
        aa1 = 1.0 + aa0  # = cc3 in R
        log_aa1 = np.log1p(aa0)
        aa2 = 1.0 / xi  # = 1/xi

        l0 = -(aa2 * (1.0 + xi) * log_aa1) - aa1 ** (-aa2) - rho
        ll = float(np.sum(l0))

        if deriv == 0:
            return {"l": ll, "l0": l0}

        # ---- First derivatives: dm, dr, dx ---
        # Precompute reused quantities (mirroring mgcv variable names)
        bb1 = sigma_inv
        bb2 = aa1  # bb1*xi*ymu+1 = aa1
        cc2 = ymu
        _cc0 = bb1 * xi * cc2  # = aa0
        log_cc3 = log_aa1
        cc3 = aa1
        dd3 = xi + 1.0
        dd6 = 1.0 / cc3
        dd7 = log_cc3
        dd8 = 1.0 / xi**2

        l1 = np.empty((n, 3), dtype=np.float64)
        # dl/dmu  (chain: dm wrt eta = 1 for identity)
        l1[:, 0] = (bb1 * dd3) / bb2 - bb1 * bb2 ** ((-1.0 / xi) - 1.0)
        # dl/drho
        l1[:, 1] = (
            (-bb1 * cc2 * cc3 ** ((-1.0 / xi) - 1.0)) + (bb1 * dd3 * cc2) / cc3 - 1.0
        )
        # dl/dxi
        aa2_v = aa2  # = 1/xi
        l1[:, 2] = (
            -(dd8 * dd7 - bb1 * aa2_v * cc2 * dd6) / cc3**aa2_v
            + dd8 * dd3 * dd7
            - aa2_v * dd7
            - bb1 * aa2_v * dd3 * cc2 * dd6
        )

        # ---- Second derivatives (6 packed: mm, mr, mx, rr, rx, xx) ----
        ee1 = np.exp(-2.0 * rho)
        ee3 = -aa2  # = -1/xi
        ff7 = ee3 - 1.0
        gg7 = -aa2
        hh4 = cc2**2
        jj08 = 1.0 / cc3**2
        jj12 = 1.0 / xi**3
        jj13 = 1.0 / cc3**aa2

        l2 = np.empty((n, 6), dtype=np.float64)
        l2[:, 0] = (
            ee1 * (ee3 - 1.0) * xi * aa1 ** (ee3 - 2.0) + (ee1 * xi * dd3) / aa1**2
        )
        l2[:, 1] = (
            bb1 * cc3**ff7
            + ee1 * ff7 * xi * cc2 * cc3 ** (ee3 - 2.0)
            - (bb1 * dd3) / cc3
            + (ee1 * xi * dd3 * cc2) / cc3**2
        )
        l2[:, 2] = (
            -bb1 * cc3 ** (gg7 - 1.0) * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
            + ee1 * cc2 * cc3 ** (gg7 - 2.0)
            + bb1 * dd6
            - (ee1 * dd3 * cc2) / cc3**2
        )
        l2[:, 3] = (
            bb1 * cc2 * cc3**ff7
            + ee1 * ff7 * xi * hh4 * cc3 ** (ee3 - 2.0)
            - (bb1 * dd3 * cc2) / cc3
            + (ee1 * xi * dd3 * hh4) / cc3**2
        )
        l2[:, 4] = (
            -bb1 * cc2 * cc3 ** (gg7 - 1.0) * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
            + ee1 * hh4 * cc3 ** (gg7 - 2.0)
            + bb1 * cc2 * dd6
            - (ee1 * dd3 * hh4) / cc3**2
        )
        l2[:, 5] = (
            -jj13 * (dd8 * dd7 - bb1 * aa2 * cc2 * dd6) ** 2
            - jj13
            * (ee1 * aa2 * hh4 * jj08 + 2.0 * bb1 * dd8 * cc2 * dd6 - 2.0 * jj12 * dd7)
            - 2.0 * jj12 * dd3 * dd7
            + 2.0 * dd8 * dd7
            + 2.0 * bb1 * dd8 * dd3 * cc2 * dd6
            - 2.0 * bb1 * aa2 * cc2 * dd6
            + ee1 * aa2 * dd3 * hh4 * jj08
        )

        # Link derivatives for chain rule
        ig1 = np.column_stack(
            [
                self.linfo[0].mu_eta(eta),
                self.linfo[1].mu_eta(etar),
                self.linfo[2].mu_eta(etax),
            ]
        )
        g2 = np.column_stack(
            [
                self.linfo[0].d2link(mu),
                self.linfo[1].d2link(rho),
                self.linfo[2].d2link(xi),
            ]
        )

        l3_val: Any = 0
        l4_val: Any = 0
        g3: Any = 0
        g4: Any = 0

        if deriv > 1:
            # Third derivatives: 10 components, order mmm mmr mmx mrr mrx mxx rrr rrx rxx xxx
            # Mirrors mgcv gamlss.r gevlss$ll deriv>1 block (auto-generated expressions).
            kk1 = np.exp(-3.0 * rho)  # 1/exp(3*rho)
            kk2 = xi**2
            ll8 = ee3 - 2.0  # -1/xi - 2
            mm11 = gg7 - 2.0  # -1/xi - 2  (same as ll8)
            mm12 = cc3**mm11  # cc3^(-1/xi - 2)
            mm10 = cc3 ** (gg7 - 3.0)  # cc3^(-1/xi - 3)
            oo10 = ff7  # gg7 - 1 = -1/xi - 1
            oo13 = log_cc3 / xi**2  # log(cc3)/xi^2
            pp08 = cc3**ff7  # cc3^(-1/xi - 1)
            qq05 = cc2**3
            rr17 = log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6
            tt08 = 1.0 / cc3**3
            tt16 = 1.0 / xi**4
            tt18 = dd8 * dd7 - bb1 * aa2 * cc2 * dd6

            l3_val = np.empty((n, 10), dtype=np.float64)
            # mmm
            l3_val[:, 0] = (2.0 * kk1 * kk2 * dd3) / cc3**3 - kk1 * (ee3 - 2.0) * (
                ee3 - 1.0
            ) * kk2 * cc3 ** (ee3 - 3.0)
            # mmr
            l3_val[:, 1] = (
                -2.0 * ee1 * ff7 * xi * cc3**ll8
                - kk1 * ll8 * ff7 * kk2 * cc2 * cc3 ** (ee3 - 3.0)
                - (2.0 * ee1 * xi * dd3) / cc3**2
                + (2.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
            )
            # mmx
            l3_val[:, 2] = (
                ee1 * ff7 * xi * mm12 * (log_cc3 / xi**2 - bb1 * aa2 * cc2 * dd6)
                - ee1 * mm12
                - kk1 * mm11 * xi * cc2 * mm10
                + kk1 * cc2 * mm10
                + ee1 * dd3 * jj08
                + ee1 * xi * jj08
                - (2.0 * kk1 * xi * dd3 * cc2) / cc3**3
            )
            # mrr
            l3_val[:, 3] = (
                -bb1 * cc3**ff7
                - 3.0 * ee1 * ff7 * xi * cc2 * cc3**ll8
                - kk1 * ll8 * ff7 * kk2 * hh4 * cc3 ** (ee3 - 3.0)
                + (bb1 * dd3) / cc3
                - (3.0 * ee1 * xi * dd3 * cc2) / cc3**2
                + (2.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
            )
            # mrx
            l3_val[:, 4] = (
                bb1 * cc3**oo10 * (bb1 * oo10 * cc2 * dd6 + oo13)
                + ee1 * oo10 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
                + ee1 * aa2 * cc2 * mm12
                + ee1 * oo10 * cc2 * mm12
                - bb1 * dd6
                + 2.0 * ee1 * dd3 * cc2 * jj08
                + ee1 * xi * cc2 * jj08
                - 2.0 * xi * dd3 * hh4 * kk1 / cc3**3
            )
            # mxx
            l3_val[:, 5] = (
                -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
                - bb1
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * dd7 / xi**3
                )
                - 2.0 * ee1 * cc2 * jj08
                + 2.0 * dd3 * hh4 * kk1 / cc3**3
            )
            # rrr
            l3_val[:, 6] = (
                -bb1 * cc2 * cc3**ff7
                - 3.0 * ee1 * ff7 * xi * hh4 * cc3**ll8
                - kk1 * ll8 * ff7 * kk2 * qq05 * cc3 ** (ee3 - 3.0)
                + (bb1 * dd3 * cc2) / cc3
                - (3.0 * ee1 * xi * dd3 * hh4) / cc3**2
                + (2.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
            )
            # rrx
            l3_val[:, 7] = (
                bb1 * cc2 * cc3**oo10 * rr17
                + ee1 * oo10 * xi * hh4 * mm12 * rr17
                - 2.0 * ee1 * hh4 * mm12
                - kk1 * mm11 * xi * qq05 * mm10
                + kk1 * qq05 * mm10
                - bb1 * cc2 * dd6
                + 2.0 * ee1 * dd3 * hh4 * jj08
                + ee1 * xi * hh4 * jj08
                - (2.0 * kk1 * xi * dd3 * qq05) / cc3**3
            )
            # rxx
            l3_val[:, 8] = (
                -bb1 * cc2 * pp08 * (bb1 * ff7 * cc2 * dd6 + dd8 * dd7) ** 2
                - bb1
                * cc2
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * dd7 / xi**3
                )
                - 2.0 * ee1 * hh4 * jj08
                + 2.0 * dd3 * qq05 * kk1 / cc3**3
            )
            # xxx
            l3_val[:, 9] = (
                -jj13 * tt18**3
                - 3.0
                * jj13
                * (
                    ee1 * aa2 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * jj12 * dd7
                )
                * tt18
                - jj13
                * (
                    -2.0 * kk1 * aa2 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * tt16 * dd7
                )
                + 6.0 * tt16 * dd3 * dd7
                - 6.0 * jj12 * dd7
                - 6.0 * bb1 * jj12 * dd3 * cc2 * dd6
                + 6.0 * bb1 * dd8 * cc2 * dd6
                - 3.0 * ee1 * dd8 * dd3 * hh4 * jj08
                + 3.0 * ee1 * aa2 * hh4 * jj08
                - 2.0 * kk1 * aa2 * dd3 * qq05 * tt08
            )
            g3 = np.column_stack(
                [
                    self.linfo[0].d3link(mu),
                    self.linfo[1].d3link(rho),
                    self.linfo[2].d3link(xi),
                ]
            )

        if deriv > 3:
            # Fourth derivatives: 15 components, order mmmm mmmr mmmx mmrr mmrx mmxx
            #                                           mrrr mrrx mrxx mxxx rrrr rrrx
            #                                           rrxx rxxx xxxx
            # Mirrors mgcv gamlss.r gevlss$ll deriv>3 block.
            # All l3 temporaries (kk1,kk2,ll8,mm11,mm12,qq05,pp08,rr17,tt08,tt16,tt18,oo13,ff7)
            # are in scope from the deriv>1 block above.
            uu1 = np.exp(-4.0 * rho)  # 1/exp(4*rho)
            uu2 = xi**3
            vv09 = ee3 - 3.0  # -1/xi - 3
            ww11 = gg7 - 3.0  # -1/xi - 3  (= vv09)
            ww12 = cc3 ** (gg7 - 4.0)  # cc3^(-1/xi - 4)
            ww15 = cc3**ww11  # cc3^(-1/xi - 3)
            # intermediate scalars reused across several l4 expressions
            ad17 = 2.0 * bb1 * dd8 * cc2 * dd6
            ad19 = -2.0 * jj12 * dd7  # -(2*dd7)/xi^3
            ad20 = pp08  # cc3^ff7
            ad21 = dd8 * dd7
            ad22 = ad21 + bb1 * mm11 * cc2 * dd6
            ae16 = dd8 * dd7 + bb1 * ff7 * cc2 * dd6
            af05 = cc2**4
            ah24 = ad19 + ad17 + ee1 * aa2 * hh4 * jj08
            aj08 = 1.0 / cc3**4
            aj20 = 1.0 / xi**5

            l4_val = np.empty((n, 15), dtype=np.float64)
            # mmmm
            l4_val[:, 0] = (
                uu1 * (ee3 - 3.0) * (ee3 - 2.0) * (ee3 - 1.0) * uu2 * cc3 ** (ee3 - 4.0)
                + (6.0 * uu1 * uu2 * dd3) / cc3**4
            )
            # mmmr
            l4_val[:, 1] = (
                3.0 * kk1 * ll8 * ff7 * kk2 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * cc2 * cc3 ** (ee3 - 4.0)
                - (6.0 * kk1 * kk2 * dd3) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * cc2) / cc3**4
            )
            # mmmx
            l4_val[:, 2] = (
                -kk1 * mm11 * ff7 * kk2 * ww15 * rr17
                + 2.0 * kk1 * mm11 * xi * ww15
                - kk1 * ww15
                + uu1 * ww11 * mm11 * kk2 * cc2 * ww12
                - uu1 * ff7 * xi * cc2 * ww12
                - uu1 * ww11 * xi * cc2 * ww12
                + 2.0 * kk1 * kk2 * tt08
                + 4.0 * kk1 * xi * dd3 * tt08
                - (6.0 * uu1 * kk2 * dd3 * cc2) / cc3**4
            )
            # mmrr
            l4_val[:, 3] = (
                4.0 * ee1 * ff7 * xi * cc3**ll8
                + 5.0 * kk1 * ll8 * ff7 * kk2 * cc2 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * hh4 * cc3 ** (ee3 - 4.0)
                + (4.0 * ee1 * xi * dd3) / cc3**2
                - (10.0 * kk1 * kk2 * dd3 * cc2) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * hh4) / cc3**4
            )
            # mmrx
            l4_val[:, 4] = (
                -2.0 * ee1 * ff7 * xi * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
                - kk1 * mm11 * ff7 * kk2 * cc2 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13)
                - 2.0 * ee1 * aa2 * mm12
                - 2.0 * ee1 * ff7 * mm12
                - 2.0 * kk1 * mm11 * ff7 * xi * cc2 * ww15
                - kk1 * ff7 * cc2 * ww15
                - kk1 * mm11 * cc2 * ww15
                - 2.0 * ee1 * dd3 * jj08
                - 2.0 * ee1 * xi * jj08
                + 2.0 * kk1 * kk2 * cc2 * tt08
                + 8.0 * kk1 * xi * dd3 * cc2 * tt08
                - 6.0 * kk2 * dd3 * hh4 * uu1 / cc3**4
            )
            # mmxx
            l4_val[:, 5] = (
                ee1 * ff7 * xi * mm12 * tt18**2
                - 2.0 * ee1 * mm12 * tt18
                - 2.0 * kk1 * mm11 * xi * cc2 * ww15 * tt18
                + 2.0 * kk1 * cc2 * ww15 * tt18
                + ee1
                * ff7
                * xi
                * mm12
                * (
                    ee1 * aa2 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * dd7 * jj12
                )
                + 4.0 * kk1 * cc2 * ww15
                + 2.0 * uu1 * ww11 * xi * hh4 * ww12
                - 4.0 * uu1 * hh4 * ww12
                + 2.0 * ee1 * jj08
                - 4.0 * kk1 * dd3 * cc2 * tt08
                - 4.0 * kk1 * xi * cc2 * tt08
                + (6.0 * uu1 * xi * dd3 * hh4) / cc3**4
            )
            # mrrr
            l4_val[:, 6] = (
                bb1 * cc3**ff7
                + 7.0 * ee1 * ff7 * xi * cc2 * cc3**ll8
                + 6.0 * kk1 * ll8 * ff7 * kk2 * hh4 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * qq05 * cc3 ** (ee3 - 4.0)
                - (bb1 * dd3) / cc3
                + (7.0 * ee1 * xi * dd3 * cc2) / cc3**2
                - (12.0 * kk1 * kk2 * dd3 * hh4) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * qq05) / cc3**4
            )
            # mrrx
            l4_val[:, 7] = (
                -bb1 * pp08 * (bb1 * ff7 * cc2 * dd6 + oo13)
                - 3.0 * ee1 * ff7 * xi * cc2 * mm12 * (bb1 * mm11 * cc2 * dd6 + oo13)
                - kk1 * mm11 * ff7 * kk2 * hh4 * ww15 * (bb1 * ww11 * cc2 * dd6 + oo13)
                - 3.0 * ee1 * aa2 * cc2 * mm12
                - 3.0 * ee1 * ff7 * cc2 * mm12
                - 2.0 * kk1 * mm11 * ff7 * xi * hh4 * ww15
                - kk1 * ff7 * hh4 * ww15
                - kk1 * mm11 * hh4 * ww15
                + bb1 * dd6
                - 4.0 * ee1 * dd3 * cc2 * jj08
                - 3.0 * ee1 * xi * cc2 * jj08
                + 2.0 * kk1 * kk2 * hh4 * tt08
                + 10.0 * kk1 * xi * dd3 * hh4 * tt08
                - 6.0 * kk2 * dd3 * qq05 * uu1 / cc3**4
            )
            # mrxx
            l4_val[:, 8] = (
                bb1 * ad20 * (bb1 * ff7 * cc2 * dd6 + ad21) ** 2
                + ee1 * ff7 * xi * cc2 * mm12 * ad22**2
                + 2.0 * ee1 * aa2 * cc2 * mm12 * ad22
                + 2.0 * ee1 * ff7 * cc2 * mm12 * ad22
                + bb1 * ad20 * (-ee1 * ff7 * hh4 * jj08 + ad17 + ad19)
                + ee1 * ff7 * xi * cc2 * mm12 * (-ee1 * mm11 * hh4 * jj08 + ad17 + ad19)
                + 4.0 * ee1 * cc2 * jj08
                - 6.0 * kk1 * dd3 * hh4 * tt08
                - 4.0 * kk1 * xi * hh4 * tt08
                + 6.0 * xi * dd3 * qq05 * uu1 / cc3**4
            )
            # mxxx
            l4_val[:, 9] = (
                -bb1 * pp08 * ae16**3
                - 3.0
                * bb1
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * jj12 * dd7
                )
                * ae16
                - bb1
                * pp08
                * (
                    2.0 * kk1 * ff7 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * dd7 * tt16
                )
                + 6.0 * kk1 * hh4 * tt08
                - 6.0 * dd3 * qq05 * uu1 / cc3**4
            )
            # rrrr
            l4_val[:, 10] = (
                bb1 * cc2 * cc3**ff7
                + 7.0 * ee1 * ff7 * xi * hh4 * cc3**ll8
                + 6.0 * kk1 * ll8 * ff7 * kk2 * qq05 * cc3**vv09
                + uu1 * vv09 * ll8 * ff7 * uu2 * af05 * cc3 ** (ee3 - 4.0)
                - (bb1 * dd3 * cc2) / cc3
                + (7.0 * ee1 * xi * dd3 * hh4) / cc3**2
                - (12.0 * kk1 * kk2 * dd3 * qq05) / cc3**3
                + (6.0 * uu1 * uu2 * dd3 * af05) / cc3**4
            )
            # rrrx
            l4_val[:, 11] = (
                -bb1 * cc2 * pp08 * rr17
                - 3.0 * ee1 * ff7 * xi * hh4 * mm12 * rr17
                - kk1 * mm11 * ff7 * kk2 * qq05 * ww15 * rr17
                + 4.0 * ee1 * hh4 * mm12
                + 5.0 * kk1 * mm11 * xi * qq05 * ww15
                - 4.0 * kk1 * qq05 * ww15
                + uu1 * ww11 * mm11 * kk2 * af05 * ww12
                - uu1 * ff7 * xi * af05 * ww12
                - uu1 * ww11 * xi * af05 * ww12
                + bb1 * cc2 * dd6
                - 4.0 * ee1 * dd3 * hh4 * jj08
                - 3.0 * ee1 * xi * hh4 * jj08
                + 2.0 * kk1 * kk2 * qq05 * tt08
                + 10.0 * kk1 * xi * dd3 * qq05 * tt08
                - 6.0 * uu1 * kk2 * dd3 * af05 / cc3**4
            )
            # rrxx
            l4_val[:, 12] = (
                bb1 * cc2 * ad20 * tt18**2
                + ee1 * ff7 * xi * hh4 * mm12 * tt18**2
                - 4.0 * ee1 * hh4 * mm12 * tt18
                - 2.0 * kk1 * mm11 * xi * qq05 * ww15 * tt18
                + 2.0 * kk1 * qq05 * ww15 * tt18
                + bb1 * cc2 * ad20 * ah24
                + ee1 * ff7 * xi * hh4 * mm12 * ah24
                + 6.0 * kk1 * qq05 * ww15
                + 2.0 * uu1 * ww11 * xi * af05 * ww12
                - 4.0 * uu1 * af05 * ww12
                + 4.0 * ee1 * hh4 * jj08
                - 6.0 * kk1 * dd3 * qq05 * tt08
                - 4.0 * kk1 * xi * qq05 * tt08
                + 6.0 * uu1 * xi * dd3 * af05 / cc3**4
            )
            # rxxx
            l4_val[:, 13] = (
                -bb1 * cc2 * pp08 * ae16**3
                - 3.0
                * bb1
                * cc2
                * pp08
                * (
                    -ee1 * ff7 * hh4 * jj08
                    + 2.0 * bb1 * dd8 * cc2 * dd6
                    - 2.0 * jj12 * dd7
                )
                * ae16
                - bb1
                * cc2
                * pp08
                * (
                    2.0 * kk1 * ff7 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * dd7 * tt16
                )
                + 6.0 * kk1 * qq05 * tt08
                - 6.0 * dd3 * af05 * uu1 / cc3**4
            )
            # xxxx
            l4_val[:, 14] = (
                -jj13 * tt18**4
                - 6.0 * jj13 * ah24 * tt18**2
                - 3.0 * jj13 * ah24**2
                - 4.0
                * jj13
                * (
                    -2.0 * kk1 * aa2 * qq05 * tt08
                    - 3.0 * ee1 * dd8 * hh4 * jj08
                    - 6.0 * bb1 * jj12 * cc2 * dd6
                    + 6.0 * tt16 * dd7
                )
                * tt18
                - jj13
                * (
                    6.0 * uu1 * aa2 * af05 * aj08
                    + 8.0 * kk1 * dd8 * qq05 * tt08
                    + 12.0 * ee1 * jj12 * hh4 * jj08
                    + 24.0 * bb1 * tt16 * cc2 * dd6
                    - 24.0 * aj20 * dd7
                )
                - 24.0 * aj20 * dd3 * dd7
                + 24.0 * tt16 * dd7
                + 24.0 * bb1 * tt16 * dd3 * cc2 * dd6
                - 24.0 * bb1 * jj12 * cc2 * dd6
                + 12.0 * ee1 * jj12 * dd3 * hh4 * jj08
                - 12.0 * ee1 * dd8 * hh4 * jj08
                + 8.0 * kk1 * dd8 * dd3 * qq05 * tt08
                - 8.0 * kk1 * aa2 * qq05 * tt08
                + 6.0 * uu1 * aa2 * dd3 * af05 * aj08
            )
            g4 = np.column_stack(
                [
                    self.linfo[0].d4link(mu),
                    self.linfo[1].d4link(rho),
                    self.linfo[2].d4link(xi),
                ]
            )

        i2 = self.tri["i2"]
        i3 = self.tri["i3"]
        i4 = self.tri["i4"]

        de = gamlss_etamu(
            l1, l2, l3_val, l4_val, ig1, g2, g3, g4, i2, i3, i4, deriv - 1
        )
        ret = gamlss_gH(
            X,
            jj,
            de["l1"],
            de["l2"],
            i2,
            l3=de["l3"],
            i3=i3,
            l4=de["l4"],
            i4=i4,
            d1b=d1b,
            d2b=d2b,
            deriv=deriv - 1,
            fh=fh,
            D=D,
            sandwich=sandwich,
        )
        if bool(kw.get("ncv", False)):
            ret["l1"] = np.asarray(de["l1"], dtype=np.float64)
            ret["l2"] = np.asarray(de["l2"], dtype=np.float64)
            ret["l3"] = de["l3"]
        ret["l"] = ll
        ret["l0"] = l0
        return ret

    def initialize(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list[np.ndarray],
        offset: Any = None,
        weights: Any = None,
        E: Any = None,
    ) -> np.ndarray:
        """
        Initialize coefficients for gevlss.

        Regress X1 on g(y) for location, X2 on log|residuals| for log-scale,
        then initialize xi near 0 (xi=1e-3 in link scale).

        Mirrors mgcv ``gevlss$initialize`` non-discrete path.
        """
        y = np.asarray(y, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape
        start = np.zeros(p, dtype=np.float64)

        offset_for_ll = offset

        use_unscaled = bool(E is not None and getattr(E, "use_unscaled", False))

        # --- Fit location predictor ---
        if self.link_names[0] == "identity":
            yt1 = y.copy()
        else:
            yt1 = self.linfo[0].linkfun(np.abs(y) + np.max(y) * 1e-7)

        X1 = X[:, jj[0]]
        if E is not None and E.shape[0] > 0:
            E1 = E[:, jj[0]]
            if use_unscaled:
                X1_aug = np.vstack([X1, E1])
                y1_aug = np.concatenate([yt1, np.zeros(E1.shape[0], dtype=np.float64)])
                start1 = _qr_coef_pivoted(X1_aug, y1_aug)
            else:
                start1 = _pen_reg(X1, E1, yt1)
        else:
            start1 = _qr_coef_pivoted(X1, yt1)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        # --- Fit log-scale predictor on log|residuals| ---
        mu_init = self.linfo[0].linkinv(X1 @ start1)
        lres1 = np.log(np.abs(y - mu_init))

        X2 = X[:, jj[1]]
        if E is not None and E.shape[0] > 0:
            E2 = E[:, jj[1]]
            if use_unscaled:
                X2_aug = np.vstack([X2, E2])
                y2_aug = np.concatenate(
                    [lres1, np.zeros(E2.shape[0], dtype=np.float64)]
                )
                start2 = _qr_coef_pivoted(X2_aug, y2_aug)
            else:
                start2 = _pen_reg(X2, E2, lres1)
        else:
            start2 = _qr_coef_pivoted(X2, lres1)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        # Mirror mgcv gevlss$initialize(): regress a constant xi start,
        # then search scalar rescalings that improve the initial log-likelihood.
        xi_init_val = 1e-3
        eta_xi0 = self.linfo[2].linkfun(np.full(1, xi_init_val))[0]
        X3 = X[:, jj[2]]
        yt3 = np.full(n, eta_xi0)

        qrx_coef = _qr_coef_pivoted(X3, yt3)
        qrx_coef = np.where(np.isfinite(qrx_coef), qrx_coef, 0.0)

        weights_arr = (
            np.ones(n, dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).ravel()
        )

        def _score_xi_scale(multiplier: float) -> tuple[float, np.ndarray]:
            start3_local = np.where(
                np.isfinite(qrx_coef * multiplier), qrx_coef * multiplier, 0.0
            )
            start_local = start.copy()
            start_local[jj[2]] = start3_local
            ll_val = float(
                self.ll(
                    y,
                    X,
                    jj,
                    start_local,
                    weights_arr,
                    offset=offset_for_ll,
                    deriv=0,
                )["l"]
            )
            return ll_val, start_local

        best_ll, best_start = _score_xi_scale(1.0)
        dm = 0.2
        mm = 1.0
        up = False
        last_ll = best_ll

        while -4.2 < mm < 4.2:
            trial_ll, trial_start = _score_xi_scale(mm + dm)
            last_ll = trial_ll
            if np.isfinite(trial_ll) and trial_ll > best_ll:
                up = True
                best_ll = trial_ll
                best_start = trial_start
                mm += dm
            elif up:
                break
            elif dm > 0.0:
                dm = -dm
            else:
                break

        if not np.isfinite(last_ll):
            trial_ll, trial_start = _score_xi_scale(mm - dm)
            if np.isfinite(trial_ll):
                best_ll = trial_ll
                best_start = trial_start

        start = best_start

        return start

    def residuals(
        self, y: np.ndarray, fitted: np.ndarray, rtype: str = "deviance"
    ) -> np.ndarray:
        """Residuals for gevlss.  Mirrors mgcv ``gevlss$residuals``."""
        rtype = str(rtype).lower()
        if rtype not in {"deviance", "pearson", "response"}:
            raise ValueError(
                "gevlss residuals support only {'deviance', 'pearson', 'response'}."
            )
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(fitted[:, 0], dtype=np.float64)
        rho = np.asarray(fitted[:, 1], dtype=np.float64)
        xi = np.asarray(fitted[:, 2], dtype=np.float64)
        sigma = np.exp(rho)
        # GEV mean: mu + sigma*(Gamma(1-xi)-1)/xi for xi != 0
        from scipy.special import gamma as gamma_fn

        fv = mu + sigma * (gamma_fn(1.0 - xi) - 1.0) / xi
        rsd = y - fv
        if rtype == "response":
            return rsd
        if rtype == "pearson":
            sd = (
                sigma / xi * np.sqrt(gamma_fn(1.0 - 2.0 * xi) - gamma_fn(1.0 - xi) ** 2)
            )
            return rsd / sd
        # deviance residuals
        eps = 1e-7
        xi2 = xi.copy()
        xi2[(xi2 >= 0) & (xi2 < eps)] = eps
        xi2[(xi2 < 0) & (xi2 > -eps)] = -eps
        aa = 1.0 + (y - mu) * np.exp(-rho) * xi2
        rsd_dev = (
            (xi2 + 1.0) / xi2 * np.log(aa)
            + aa ** (-1.0 / xi2)
            + (1.0 + xi2) * np.log(1.0 + xi2)
            - (1.0 + xi2)
        )
        return np.sqrt(np.maximum(0.0, rsd_dev)) * np.sign(y - fv)


def gevlss(link=("identity", "identity", "logit")) -> GevlssFamily:
    """
    Generalized Extreme Value location-scale-shape family.

    Parameters
    ----------
    link : tuple of str
        Links for (location, log-scale, shape).
        Location: ``"identity"`` (default) or ``"log"``.
        Log-scale: ``"identity"`` only.
        Shape xi ∈ (-1, 0.5): ``"logit"`` (default, shifted) or ``"identity"``.

    Returns
    -------
    GevlssFamily instance.
    """
    return GevlssFamily(link=link)


# ---------------------------------------------------------------------------
# shashlss: Sinh-arcsinh location-scale-skewness-kurtosis  (mgcv: shash)
# ---------------------------------------------------------------------------
