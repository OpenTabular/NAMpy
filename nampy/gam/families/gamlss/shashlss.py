from __future__ import annotations

import numpy as np
from scipy.special import kv

from ...fit.solvers.gamlss_utils import gamlss_etamu, gamlss_gH, trind_generator
from ._base import GamlssFamily, _IdentityLinkInfo, _pen_reg, _qr_coef_pivoted


class _LogEBLinkInfo:
    """
    'logeb' link for shash log-scale parameter tau:
        linkinv(eta) = log(exp(eta) + b)   →   tau = log(exp(eta) + b)
        linkfun(tau) = log(exp(tau) - b)

    Mirrors the logeb link constructed inside mgcv ``shash(b=0.01)``.
    """

    name = "logeb"

    def __init__(self, b: float = 0.01):
        self.b = float(b)

    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        return np.log(np.exp(mu) - self.b)

    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        return np.log(np.exp(eta) + self.b)

    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        eta = np.asarray(eta, dtype=np.float64)
        ee = np.exp(eta)
        return ee / (ee + self.b)

    def d2link(self, mu: np.ndarray) -> np.ndarray:
        # d^2 eta / d mu^2 = fr*(1-fr) where fr = exp(mu)/(exp(mu)-b)
        mu = np.asarray(mu, dtype=np.float64)
        em = np.exp(mu)
        fr = em / (em - self.b)
        return fr * (1.0 - fr)

    def d3link(self, mu: np.ndarray) -> np.ndarray:
        # d^3 eta / d mu^3 = oo - 2*oo*fr  (oo = fr*(1-fr))
        mu = np.asarray(mu, dtype=np.float64)
        em = np.exp(mu)
        fr = em / (em - self.b)
        oo = fr * (1.0 - fr)
        return oo - 2.0 * oo * fr

    def d4link(self, mu: np.ndarray) -> np.ndarray:
        # -b*em*(b^2 + 4*b*em + em^2) / (em - b)^4
        mu = np.asarray(mu, dtype=np.float64)
        em = np.exp(mu)
        denom = (em - self.b) ** 4
        return -self.b * em * (self.b**2 + 4.0 * self.b * em + em**2) / denom


class ShashlssFamily(GamlssFamily):
    """
    Sinh-arcsinh (shash) family with four linear predictors:
      1. location μ        (identity link)
      2. log-scale τ       (logeb link, lower-bounded)
      3. skewness ε        (identity link)
      4. log-kurtosis φ    (identity link)

    Derived quantities: sig = exp(τ), del = exp(φ).
    Observation model: Y = μ + sig*del*sinh((1/del)*asinh(Z) + ε/del)
    where Z ~ N(0,1).

    Mirrors mgcv ``shash`` (Fasiolo 2020) with phiPen=1e-3.

    Note: raw ``ll(..., deriv=4)`` blocks now support analytic outer gradient
    and Hessian terms through ``gam.fit5``.
    """

    name = "shashlss"
    nlp = 4
    n_linear_predictors = 4
    supports_analytic_outer_derivatives = False
    supports_analytic_outer_gradient = True
    supports_analytic_outer_hessian = True

    def __init__(self, b: float = 0.01, phi_pen: float = 1e-3):
        self.b = float(b)
        self.phi_pen = float(phi_pen)
        self.linfo = [
            _IdentityLinkInfo(),  # mu
            _LogEBLinkInfo(b=b),  # tau = log(sigma)
            _IdentityLinkInfo(),  # eps (skewness)
            _IdentityLinkInfo(),  # phi = log(delta) (kurtosis)
        ]
        self.tri = trind_generator(4)

    @staticmethod
    def _sqrtX2pm(x: np.ndarray, m: float) -> np.ndarray:
        """sqrt(x^2 + m), stable for large |x|.  Mirrors mgcv .sqrtX2pm."""
        x = np.abs(x)
        out = x.copy()
        small = x < 1e8
        out[small] = np.sqrt(x[small] ** 2 + m)
        return out

    @staticmethod
    def _ax2m1DivX2m2SQ(
        x: np.ndarray, m1: float, m2: float, a: float = 1.0
    ) -> np.ndarray:
        """(a*x^2 + m1) / (x^2 + m2)^2, stable for large |x|.

        Mirrors mgcv .ax2m1DivX2m2SQ.  Called with m1=-1, m2=1, a=1 to
        compute (z^2-1)/(z^2+1)^2 in Dmm.
        """
        x = np.abs(x)
        out = np.zeros_like(x, dtype=np.float64)
        neg = (a * x**2 + m1) < 0.0
        if np.any(neg):
            xn = x[neg]
            out[neg] = (a * xn**2 + m1) / (xn**2 + m2) ** 2
        pos = ~neg
        if np.any(pos):
            xp = x[pos]
            sq_num = ShashlssFamily._sqrtX2pm(np.sqrt(a) * xp, m1)
            sq_den = ShashlssFamily._sqrtX2pm(xp, m2)
            out[pos] = (sq_num / sq_den / sq_den) ** 2
        return out

    def ll(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list,
        coef: np.ndarray,
        weights: np.ndarray,
        offset=None,
        deriv: int = 0,
        d1b=0,
        d2b=0,
        fh=None,
        D=None,
        **kw,
    ) -> dict:
        """Log-likelihood and derivatives for shash.  Mirrors mgcv shash$ll."""
        y = np.asarray(y, dtype=np.float64)
        coef = np.asarray(coef, dtype=np.float64)
        sandwich = bool(kw.get("sandwich", False))
        offsets = self._offset_list(offset)
        if any(
            off is not None and np.sum(np.abs(np.asarray(off, dtype=np.float64))) != 0.0
            for off in offsets
        ):
            raise NotImplementedError("mgcv shash does not support non-zero offsets.")
        if weights is None:
            weights = np.ones(len(y), dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        eta_mat = self._eta_matrix_from_inputs(
            X,
            jj,
            coef,
            offset=offset,
            eta=kw.get("eta", None),
        )
        eta = np.asarray(eta_mat[:, 0], dtype=np.float64)
        eta1 = np.asarray(eta_mat[:, 1], dtype=np.float64)
        eta2 = np.asarray(eta_mat[:, 2], dtype=np.float64)
        eta3 = np.asarray(eta_mat[:, 3], dtype=np.float64)

        mu = self.linfo[0].linkinv(eta)
        tau = self.linfo[1].linkinv(eta1)
        eps = self.linfo[2].linkinv(eta2)
        phi = self.linfo[3].linkinv(eta3)

        sig = np.exp(tau)
        delta = np.exp(phi)

        z = (y - mu) / (sig * delta)
        dTasMe = delta * np.arcsinh(z) - eps
        g = -dTasMe
        CC = np.cosh(dTasMe)
        SS = np.sinh(dTasMe)

        log1pz2 = np.log1p(np.exp(2.0 * np.log(np.abs(z))))
        l0 = (
            -tau
            - 0.5 * np.log(2.0 * np.pi)
            + np.log(np.maximum(CC, 1e-300))
            - 0.5 * log1pz2
            - 0.5 * SS**2
            - self.phi_pen * phi**2
        )
        ll = float(np.sum(l0))

        if not deriv:
            return {"l": ll, "l0": l0}

        # ----------------------------------------------------------------
        # First derivatives w.r.t. distribution parameters (mu,tau,eps,phi)
        # ----------------------------------------------------------------
        zsd = y - mu  # = z * sig * delta
        sSp1 = self._sqrtX2pm(z, 1.0)  # sqrt(z^2 + 1)
        asinhZ = np.arcsinh(z)

        tanh_g = np.tanh(g)
        sinh_2g = np.sinh(2.0 * g)
        sech_g_sq = 1.0 / np.cosh(g) ** 2

        De = tanh_g - 0.5 * sinh_2g
        Dm = (delta * De + z / sSp1) / (delta * sig * sSp1)
        Dt = zsd * Dm - 1.0
        Dp = Dt + 1.0 - delta * asinhZ * De - 2.0 * self.phi_pen * phi

        # shape: (n, 4) — L1 in mgcv
        L1 = np.column_stack([Dm, Dt, De, Dp])

        # ----------------------------------------------------------------
        # Second derivatives
        # ----------------------------------------------------------------
        cosh_2g = np.cosh(2.0 * g)

        Dme = (sech_g_sq - cosh_2g) / (sig * sSp1)
        Dte = zsd * Dme
        Dmm = (
            Dme / (sig * sSp1)
            + z * De / (sig**2 * delta * sSp1**3)
            + self._ax2m1DivX2m2SQ(z, -1.0, 1.0) / (delta * sig) ** 2
        )
        Dmt = zsd * Dmm - Dm
        Dee = -2.0 * np.cosh(g) ** 2 + sech_g_sq + 1.0
        Dtt = zsd * Dmt
        Dep = Dte - delta * asinhZ * Dee
        Dmp = Dmt + De / (sig * sSp1) - delta * asinhZ * Dme
        Dtp = zsd * Dmp
        Dpp = (
            Dtp
            - delta * asinhZ * Dep
            + delta * (z / sSp1 - asinhZ) * De
            - 2.0 * self.phi_pen
        )

        # shape: (n, 10) — L2 in mgcv (packed upper-triangular)
        L2 = np.column_stack([Dmm, Dmt, Dme, Dmp, Dtt, Dte, Dtp, Dee, Dep, Dpp])

        # Link derivatives for chain rule
        IG1 = np.column_stack(
            [
                self.linfo[0].mu_eta(eta),
                self.linfo[1].mu_eta(eta1),
                self.linfo[2].mu_eta(eta2),
                self.linfo[3].mu_eta(eta3),
            ]
        )
        G2 = np.column_stack(
            [
                self.linfo[0].d2link(mu),
                self.linfo[1].d2link(tau),
                self.linfo[2].d2link(eps),
                self.linfo[3].d2link(phi),
            ]
        )

        L3 = G3 = L4 = G4 = 0

        if deriv > 1:
            # Third derivatives: 20 components for K=4 predictors.
            # Order: Dmmm Dmmt Dmme Dmmp Dmtt Dmte Dmtp Dmee Dmep Dmpp
            #        Dttt Dtte Dttp Dtee Dtep Dtpp Deee Deep Depp Dppp
            # Mirrors mgcv gamlss.r shash$ll deriv>1 block.
            Deee = -2.0 * (sinh_2g + sech_g_sq * tanh_g)
            Dmee = Deee / (sig * sSp1)
            Dmme = Dmee / (sig * sSp1) + z * Dee / (sig**2 * delta * sSp1**3)
            Dmmm = (
                2.0 * z * Dme / (sig**2 * delta * sSp1**3)
                + Dmme / (sig * sSp1)
                + self._ax2m1DivX2m2SQ(z, -1.0, 1.0, 2.0)
                * De
                / (sig**3 * delta**2 * sSp1)
                + 2.0
                * (z / sSp1)
                * self._ax2m1DivX2m2SQ(z, -3.0, 1.0)
                / ((sig * delta) ** 3 * sSp1)
            )
            Dmmt = zsd * Dmmm - 2.0 * Dmm
            Dtee = zsd * Dmee
            Dmte = zsd * Dmme - Dme
            Dtte = zsd * Dmte
            Dmtt = zsd * Dmmt - Dmt
            Dttt = zsd * Dmtt
            Dmep = Dmte + Dee / (sig * sSp1) - delta * asinhZ * Dmee
            Dtep = zsd * Dmep
            Deep = Dtee - delta * asinhZ * Deee
            Depp = Dtep - delta * asinhZ * Deep + delta * (z / sSp1 - asinhZ) * Dee
            Dmmp = (
                Dmmt
                + 2.0 * Dme / (sig * sSp1)
                + z * De / (delta * sig**2 * sSp1**3)
                - delta * asinhZ * Dmme
            )
            Dmtp = zsd * Dmmp - Dmp
            Dttp = zsd * Dmtp
            Dmpp = (
                Dmtp
                + Dep / (sig * sSp1)
                + z**2 * De / (sig * sSp1**3)
                - delta * asinhZ * Dmep
                + delta * Dme * (z / sSp1 - asinhZ)
            )
            Dtpp = zsd * Dmpp
            Dppp = (
                Dtpp
                - delta * asinhZ * Depp
                + delta * (z / sSp1 - asinhZ) * (2.0 * Dep + De)
                + delta * (z / sSp1) ** 3 * De
            )
            L3 = np.column_stack(
                [
                    Dmmm,
                    Dmmt,
                    Dmme,
                    Dmmp,
                    Dmtt,
                    Dmte,
                    Dmtp,
                    Dmee,
                    Dmep,
                    Dmpp,
                    Dttt,
                    Dtte,
                    Dttp,
                    Dtee,
                    Dtep,
                    Dtpp,
                    Deee,
                    Deep,
                    Depp,
                    Dppp,
                ]
            )
            G3 = np.column_stack(
                [
                    self.linfo[0].d3link(mu),
                    self.linfo[1].d3link(tau),
                    self.linfo[2].d3link(eps),
                    self.linfo[3].d3link(phi),
                ]
            )

        if deriv > 3:
            # 35 fourth-order derivatives from mgcv gamlss.r shashlss$ll (deriv>3).
            # R aliases: m→mu, t→tau, p→phi, e→eps, aaa2→zsd; exp1^x→np.exp(x); ^→**.
            abb8 = CC
            abb9 = SS
            abb1 = np.exp(-2.0 * tau - 2.0 * phi)
            abb3 = zsd**2
            abb4 = np.exp(-tau)
            abb5 = -tau - phi
            abb7 = np.exp(2.0 * abb5) * abb3 + 1.0
            abb6 = 1.0 / np.sqrt(abb7)
            aee5 = dTasMe + eps
            aff04 = abb1 * abb3 + 1.0
            aff05 = abb4**2
            aff08 = 2.0 * abb5
            aff10 = 1.0 / abb7
            aff13 = CC**2
            aff14 = np.exp(-tau + aff08)
            aff15 = abb6**3
            aff17 = SS**2
            agg15 = 1.0 / abb6
            agg17 = 1.0 / CC
            aii11 = dTasMe + eps
            aii12 = aii11 - abb4 * zsd * abb6
            aii17 = abb6**3
            ajj15 = zsd**3
            ann05 = np.exp(phi)
            ann06 = np.arcsinh(np.exp(abb5) * zsd)
            aoo09 = -zsd / (np.exp(tau) * agg15)
            app04 = np.exp(-2.0 * tau - 2.0 * phi) * abb3 + 1.0
            app08 = np.exp(-2.0 * tau + aff08)
            app10 = 1.0 / abb7**2
            app14 = np.exp(-tau + 4.0 * abb5)
            app16 = 1.0 / agg15**5
            app21 = 1.0 / np.exp(3.0 * tau)
            aqq03 = np.exp(-2.0 * tau - 2.0 * phi)
            aqq05 = aqq03 * abb3 + 1.0
            aqq27 = 1.0 / aff13
            arr06 = np.exp(aff08) * zsd**2 + 1.0
            arr07 = 1.0 / np.sqrt(arr06) ** 3
            arr12 = 1.0 / arr06
            ass16 = aii11 - zsd / (np.exp(tau) * agg15)
            ass23 = 1.0 / CC
            ass28 = 1.0 / aff13
            att19 = zsd**4
            avv19 = aii11 - abb4 * zsd * abb6
            ayy14 = -abb4 * zsd * abb6
            ayy16 = aii11 + ayy14
            ayy17 = aii11 + ayy14 - aff14 * ajj15 * aii17
            ayy24 = ayy16**2
            azz19 = zsd**5
            bdd07 = np.sqrt(np.exp(aff08) * zsd**2 + 1.0)
            bdd08 = 1.0 / bdd07**3
            bdd14 = 1.0 / bdd07
            bdd15 = aii11 - abb4 * zsd * bdd14
            bgg4 = aee5 - zsd / (
                np.exp(tau) * np.sqrt(np.exp(2.0 * abb5) * zsd**2 + 1.0)
            )
            bhh13 = -abb4 * zsd * bdd14
            bhh14 = ann05 * ann06
            bii11 = aii11 + aoo09
            bii15 = aii11 + aoo09 - aff14 * ajj15 * aii17
            bjj07 = 4.0 * abb5
            bjj08 = np.exp(-2.0 * tau + bjj07)
            bjj11 = 1.0 / abb7**3
            bjj14 = 1.0 / np.exp(4.0 * tau)
            bjj18 = np.exp(-tau + 6.0 * abb5)
            bjj21 = 1.0 / agg15**7
            bjj24 = np.exp(aff08 - 3.0 * tau)
            bjj26 = np.exp(-tau + bjj07)
            bkk33 = 1.0 / CC**3
            bkk34 = SS**3
            bll16 = np.exp(aff08 - 2.0 * tau)
            bmm34 = 1.0 / CC**3
            bmm35 = SS**3
            bss21 = 2.0 * aff14 * abb3 * aff15 - 3.0 * bjj26 * att19 * app16
            bss23 = -abb4 * zsd * abb6
            bss25 = aii11 + bss23
            bss26 = aii11 + bss23 - aff14 * ajj15 * aff15
            bss29 = bss25**2
            bss33 = (
                -4.0 * aff14 * zsd * aff15
                + 18.0 * bjj26 * ajj15 * app16
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**5 / agg15**7
            )
            btt24 = zsd**6
            byy24 = 2.0 * aff14 * ajj15 * aff15 - 3.0 * bjj26 * azz19 * app16
            byy35 = (
                -6.0 * aff14 * abb3 * aff15
                + 21.0 * bjj26 * att19 * app16
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**6 / agg15**7
            )
            bzz7 = CC**2
            bzz9 = SS**2
            cbb09 = 1.0 / agg15**5
            cbb18 = 2.0 * aff14 * abb3 * aii17 - 3.0 * app14 * att19 * cbb09
            cbb24 = aii11 + ayy14 - aff14 * zsd**3 * aii17
            cdd24 = zsd**7
            cll08 = 1.0 / bdd07**5
            cll16 = aii11 + bhh13
            cll17 = cll16**2
            cll18 = 2.0 * aff14 * ajj15 * bdd08 - 3.0 * app14 * azz19 * cll08
            cll24 = aii11 + bhh13 - aff14 * ajj15 * bdd08
            cmm12 = -3.0 * app14 * azz19 * cbb09
            cmm16 = 2.0 * aff14 * ajj15 * aii17 + cmm12
            cmm23 = aii11 + ayy14 + aff14 * ajj15 * aii17 + cmm12
            cmm28 = (
                -4.0 * aff14 * ajj15 * aii17
                + 18.0 * app14 * azz19 * cbb09
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**7 / agg15**7
            )
            cpp06 = -zsd / (np.exp(tau) * bdd07)
            cpp08 = (cpp06 + aii11) ** 2
            cpp12 = aii11 + cpp06 - np.exp(-tau + aff08) * zsd**3 / bdd07**3
            cqq12 = -aff14 * ajj15 * bdd08
            cqq19 = bhh14 + bhh13
            cqq20 = cqq19**3
            cqq21 = bhh14 + bhh13 + aff14 * ajj15 * bdd08 - 3.0 * app14 * azz19 * cll08
            cqq25 = bhh14 + bhh13 + cqq12
            cqq28 = 1.0 / aff13
            crr18 = aii11 + aoo09 + aff14 * ajj15 * aii17 - 3.0 * app14 * azz19 * cbb09
            crr19 = bii11**4
            crr21 = bii15**2
            crr25 = (
                aii11
                + aoo09
                - 3.0 * aff14 * ajj15 * aii17
                + 15.0 * app14 * azz19 * cbb09
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**7 / agg15**7
            )
            crr28 = bii11**2
            ccc23 = aii11 + ayy14 + aff14 * ajj15 * aii17 - 3.0 * app14 * azz19 * cbb09
            ccc24 = ayy16**3
            ccc28 = (
                -4.0 * aff14 * abb3 * aii17
                + 18.0 * app14 * att19 * cbb09
                - 15.0 * np.exp(-tau + 6.0 * abb5) * zsd**6 / agg15**7
            )
            cnn3 = CC**2
            cnn5 = SS**2
            coo7 = CC**2
            coo9 = SS**2

            # j2 (mmmm)
            j2 = (
                -(6.0 * bjj14 * app10 * abb9**4) / abb8**4
                - (12.0 * bjj24 * zsd * app16 * abb9**3) / abb8**3
                + 8.0 * bjj14 * app10 * aqq27 * aff17
                + 4.0 * app08 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * abb3 * bjj11 * aqq27 * aff17
                - 4.0 * bjj14 * app10 * aff17
                + 4.0 * app08 * app10 * aff17
                - 15.0 * bjj08 * abb3 * bjj11 * aff17
                - 9.0 * bjj26 * zsd * app16 * abb8 * abb9
                + 24.0 * bjj24 * zsd * app16 * abb8 * abb9
                + 15.0 * bjj18 * ajj15 * bjj21 * abb8 * abb9
                + 9.0 * bjj26 * zsd * app16 * agg17 * abb9
                + 12.0 * bjj24 * zsd * app16 * agg17 * abb9
                - 15.0 * bjj18 * ajj15 * bjj21 * agg17 * abb9
                - 4.0 * bjj14 * app10 * aff13
                + 4.0 * app08 * app10 * aff13
                - 15.0 * bjj08 * abb3 * bjj11 * aff13
                - 2.0 * bjj14 * app10
                - 4.0 * app08 * app10
                + 15.0 * bjj08 * abb3 * bjj11
                + (6.0 * np.exp((-4.0 * tau) - 4.0 * phi)) / app04**2
                - (48.0 * np.exp((-6.0 * tau) - 6.0 * phi) * abb3) / app04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**4) / app04**4
            )
            # k2 (mmmt)
            k2 = (
                -(6.0 * bjj14 * zsd * app10 * abb9**4) / abb8**4
                + 6.0 * app21 * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * abb3 * app16 * bkk33 * bkk34
                + 8.0 * bjj14 * zsd * app10 * aqq27 * aff17
                + 13.0 * app08 * zsd * app10 * aqq27 * aff17
                - 15.0 * bjj08 * ajj15 * bjj11 * aqq27 * aff17
                - 4.0 * bjj14 * zsd * app10 * aff17
                + 13.0 * app08 * zsd * app10 * aff17
                - 15.0 * bjj08 * ajj15 * bjj11 * aff17
                - 12.0 * app21 * aff15 * abb8 * abb9
                + 3.0 * aff14 * aff15 * abb8 * abb9
                - 18.0 * bjj26 * abb3 * app16 * abb8 * abb9
                + 24.0 * bjj24 * abb3 * app16 * abb8 * abb9
                + 15.0 * bjj18 * att19 * bjj21 * abb8 * abb9
                - 6.0 * app21 * aff15 * agg17 * abb9
                - 3.0 * aff14 * aff15 * agg17 * abb9
                + 18.0 * bjj26 * abb3 * app16 * agg17 * abb9
                + 12.0 * bjj24 * abb3 * app16 * agg17 * abb9
                - 15.0 * bjj18 * att19 * bjj21 * agg17 * abb9
                - 4.0 * bjj14 * zsd * app10 * aff13
                + 13.0 * app08 * zsd * app10 * aff13
                - 15.0 * bjj08 * ajj15 * bjj11 * aff13
                - 2.0 * bjj14 * zsd * app10
                - 13.0 * app08 * zsd * app10
                + 15.0 * bjj08 * ajj15 * bjj11
                + (24.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd) / app04**2
                - (72.0 * np.exp((-6.0 * tau) - 6.0 * phi) * ajj15) / app04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**5) / app04**4
            )
            # l2 (mmme)
            l2 = (
                -(6.0 * app21 * aff15 * abb9**4) / abb8**4
                - (6.0 * bll16 * zsd * app10 * abb9**3) / abb8**3
                + 8.0 * app21 * aff15 * aqq27 * aff17
                + aff14 * aff15 * aqq27 * aff17
                - 3.0 * app14 * abb3 * app16 * aqq27 * aff17
                - 4.0 * app21 * aff15 * aff17
                + aff14 * aff15 * aff17
                - 3.0 * app14 * abb3 * app16 * aff17
                + 12.0 * bll16 * zsd * app10 * abb8 * abb9
                + (6.0 * bll16 * zsd * app10 * abb9) / abb8
                - 4.0 * app21 * aff15 * aff13
                + aff14 * aff15 * aff13
                - 3.0 * app14 * abb3 * app16 * aff13
                - 2.0 * app21 * aff15
                - aff14 * aff15
                + 3.0 * app14 * abb3 * app16
            )
            # m2 (mmmp)
            m2 = (
                (6.0 * app21 * aff15 * ass16 * abb9**4) / abb8**4
                + 6.0 * app08 * zsd * app10 * ass16 * bmm34 * bmm35
                - 6.0 * bjj24 * abb3 * app16 * bmm34 * bmm35
                - 8.0 * app21 * aff15 * ass16 * ass28 * aff17
                - aff14 * aff15 * ass16 * ass28 * aff17
                + 3.0 * bjj26 * abb3 * app16 * ass16 * ass28 * aff17
                + 6.0 * app08 * zsd * app10 * ass28 * aff17
                - 12.0 * bjj08 * ajj15 * bjj11 * ass28 * aff17
                + 4.0 * app21 * aff15 * ass16 * aff17
                - aff14 * aff15 * ass16 * aff17
                + 3.0 * bjj26 * abb3 * app16 * ass16 * aff17
                + 6.0 * app08 * zsd * app10 * aff17
                - 12.0 * bjj08 * ajj15 * bjj11 * aff17
                - 12.0 * app08 * zsd * app10 * ass16 * abb8 * abb9
                + 2.0 * aff14 * aff15 * abb8 * abb9
                - 15.0 * bjj26 * abb3 * app16 * abb8 * abb9
                + 12.0 * bjj24 * abb3 * app16 * abb8 * abb9
                + 15.0 * bjj18 * att19 * bjj21 * abb8 * abb9
                - 6.0 * app08 * zsd * app10 * ass16 * ass23 * abb9
                - 2.0 * aff14 * aff15 * ass23 * abb9
                + 15.0 * bjj26 * abb3 * app16 * ass23 * abb9
                + 6.0 * bjj24 * abb3 * app16 * ass23 * abb9
                - 15.0 * bjj18 * att19 * bjj21 * ass23 * abb9
                + 4.0 * app21 * aff15 * ass16 * aff13
                - aff14 * aff15 * ass16 * aff13
                + 3.0 * bjj26 * abb3 * app16 * ass16 * aff13
                + 6.0 * app08 * zsd * app10 * aff13
                - 12.0 * bjj08 * ajj15 * bjj11 * aff13
                + 2.0 * app21 * aff15 * ass16
                + aff14 * aff15 * ass16
                - 3.0 * bjj26 * abb3 * app16 * ass16
                - 6.0 * app08 * zsd * app10
                + 12.0 * bjj08 * ajj15 * bjj11
                + (24.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd) / app04**2
                - (72.0 * np.exp((-6.0 * tau) - 6.0 * phi) * ajj15) / app04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**5) / app04**4
            )
            # n2 (mmtt)
            n2 = (
                -(6.0 * bjj14 * abb3 * app10 * abb9**4) / abb8**4
                + 10.0 * app21 * zsd * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * ajj15 * app16 * bkk33 * bkk34
                - 4.0 * aff05 * aff10 * aqq27 * aff17
                + 8.0 * bjj14 * abb3 * app10 * aqq27 * aff17
                + 19.0 * app08 * abb3 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * att19 * bjj11 * aqq27 * aff17
                - 4.0 * aff05 * aff10 * aff17
                - 4.0 * bjj14 * abb3 * app10 * aff17
                + 19.0 * app08 * abb3 * app10 * aff17
                - 15.0 * bjj08 * att19 * bjj11 * aff17
                - 20.0 * app21 * zsd * aff15 * abb8 * abb9
                + 9.0 * aff14 * zsd * aff15 * abb8 * abb9
                - 24.0 * bjj26 * ajj15 * app16 * abb8 * abb9
                + 24.0 * bjj24 * ajj15 * app16 * abb8 * abb9
                + 15.0 * bjj18 * azz19 * bjj21 * abb8 * abb9
                - 10.0 * app21 * zsd * aff15 * agg17 * abb9
                - 9.0 * aff14 * zsd * aff15 * agg17 * abb9
                + 24.0 * bjj26 * ajj15 * app16 * agg17 * abb9
                + 12.0 * bjj24 * ajj15 * app16 * agg17 * abb9
                - 15.0 * bjj18 * azz19 * bjj21 * agg17 * abb9
                - 4.0 * aff05 * aff10 * aff13
                - 4.0 * bjj14 * abb3 * app10 * aff13
                + 19.0 * app08 * abb3 * app10 * aff13
                - 15.0 * bjj08 * att19 * bjj11 * aff13
                + 4.0 * aff05 * aff10
                - 2.0 * bjj14 * abb3 * app10
                - 19.0 * app08 * abb3 * app10
                + 15.0 * bjj08 * att19 * bjj11
                - (4.0 * aqq03) / aqq05
                + (44.0 * np.exp((-4.0 * tau) - 4.0 * phi) * abb3) / aqq05**2
                - (88.0 * np.exp((-6.0 * tau) - 6.0 * phi) * att19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**6) / aqq05**4
            )
            # o2 (mmte)
            o2 = (
                -(6.0 * app21 * zsd * aff15 * abb9**4) / abb8**4
                + 4.0 * aff05 * aff10 * bkk33 * bkk34
                - 6.0 * bll16 * abb3 * app10 * bkk33 * bkk34
                + 8.0 * app21 * zsd * aff15 * aqq27 * aff17
                + 3.0 * aff14 * zsd * aff15 * aqq27 * aff17
                - 3.0 * app14 * ajj15 * app16 * aqq27 * aff17
                - 4.0 * app21 * zsd * aff15 * aff17
                + 3.0 * aff14 * zsd * aff15 * aff17
                - 3.0 * app14 * ajj15 * app16 * aff17
                - 8.0 * aff05 * aff10 * abb8 * abb9
                + 12.0 * bll16 * abb3 * app10 * abb8 * abb9
                - 4.0 * aff05 * aff10 * agg17 * abb9
                + 6.0 * bll16 * abb3 * app10 * agg17 * abb9
                - 4.0 * app21 * zsd * aff15 * aff13
                + 3.0 * aff14 * zsd * aff15 * aff13
                - 3.0 * app14 * ajj15 * app16 * aff13
                - 2.0 * app21 * zsd * aff15
                - 3.0 * aff14 * zsd * aff15
                + 3.0 * app14 * ajj15 * app16
            )
            # p2 (mmtp)
            p2 = (
                (6.0 * app21 * zsd * aff15 * ass16 * abb9**4) / abb8**4
                - 4.0 * aff05 * aff10 * ass16 * bmm34 * bmm35
                + 6.0 * app08 * abb3 * app10 * ass16 * bmm34 * bmm35
                - 6.0 * bjj24 * ajj15 * app16 * bmm34 * bmm35
                - 8.0 * app21 * zsd * aff15 * ass16 * ass28 * aff17
                - 3.0 * aff14 * zsd * aff15 * ass16 * ass28 * aff17
                + 3.0 * bjj26 * ajj15 * app16 * ass16 * ass28 * aff17
                + 10.0 * app08 * abb3 * app10 * ass28 * aff17
                - 12.0 * bjj08 * att19 * bjj11 * ass28 * aff17
                + 4.0 * app21 * zsd * aff15 * ass16 * aff17
                - 3.0 * aff14 * zsd * aff15 * ass16 * aff17
                + 3.0 * bjj26 * ajj15 * app16 * ass16 * aff17
                + 10.0 * app08 * abb3 * app10 * aff17
                - 12.0 * bjj08 * att19 * bjj11 * aff17
                + 8.0 * aff05 * aff10 * ass16 * abb8 * abb9
                - 12.0 * app08 * abb3 * app10 * ass16 * abb8 * abb9
                + 6.0 * aff14 * zsd * aff15 * abb8 * abb9
                - 21.0 * bjj26 * ajj15 * app16 * abb8 * abb9
                + 12.0 * bjj24 * ajj15 * app16 * abb8 * abb9
                + 15.0 * bjj18 * azz19 * bjj21 * abb8 * abb9
                + 4.0 * aff05 * aff10 * ass16 * ass23 * abb9
                - 6.0 * app08 * abb3 * app10 * ass16 * ass23 * abb9
                - 6.0 * aff14 * zsd * aff15 * ass23 * abb9
                + 21.0 * bjj26 * ajj15 * app16 * ass23 * abb9
                + 6.0 * bjj24 * ajj15 * app16 * ass23 * abb9
                - 15.0 * bjj18 * azz19 * bjj21 * ass23 * abb9
                + 4.0 * app21 * zsd * aff15 * ass16 * aff13
                - 3.0 * aff14 * zsd * aff15 * ass16 * aff13
                + 3.0 * bjj26 * ajj15 * app16 * ass16 * aff13
                + 10.0 * app08 * abb3 * app10 * aff13
                - 12.0 * bjj08 * att19 * bjj11 * aff13
                + 2.0 * app21 * zsd * aff15 * ass16
                + 3.0 * aff14 * zsd * aff15 * ass16
                - 3.0 * bjj26 * ajj15 * app16 * ass16
                - 10.0 * app08 * abb3 * app10
                + 12.0 * bjj08 * att19 * bjj11
                - (4.0 * aqq03) / aqq05
                + (44.0 * np.exp((-4.0 * tau) - 4.0 * phi) * abb3) / aqq05**2
                - (88.0 * np.exp((-6.0 * tau) - 6.0 * phi) * att19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**6) / aqq05**4
            )
            # q2 (mmee)
            q2 = (
                -(6.0 * aff05 * arr12 * abb9**4) / abb8**4
                - (2.0 * aff14 * zsd * arr07 * abb9**3) / abb8**3
                + (8.0 * aff05 * arr12 * aff17) / aff13
                - 4.0 * aff05 * arr12 * aff17
                + 4.0 * aff14 * zsd * arr07 * abb8 * abb9
                + (2.0 * aff14 * zsd * arr07 * abb9) / abb8
                - 4.0 * aff05 * arr12 * aff13
                - 2.0 * aff05 * arr12
            )
            # r2 (mmep)
            r2 = (
                (6.0 * aff05 * aff10 * ass16 * abb9**4) / abb8**4
                + 2.0 * aff14 * zsd * aff15 * ass16 * bmm34 * bmm35
                - 4.0 * bll16 * abb3 * app10 * bmm34 * bmm35
                - 8.0 * aff05 * aff10 * ass16 * ass28 * aff17
                + 2.0 * aff14 * zsd * aff15 * ass28 * aff17
                - 3.0 * app14 * ajj15 * app16 * ass28 * aff17
                + 4.0 * aff05 * aff10 * ass16 * aff17
                + 2.0 * aff14 * zsd * aff15 * aff17
                - 3.0 * app14 * ajj15 * app16 * aff17
                - 4.0 * aff14 * zsd * aff15 * ass16 * abb8 * abb9
                + 8.0 * bll16 * abb3 * app10 * abb8 * abb9
                - 2.0 * aff14 * zsd * aff15 * ass16 * ass23 * abb9
                + 4.0 * bll16 * abb3 * app10 * ass23 * abb9
                + 4.0 * aff05 * aff10 * ass16 * aff13
                + 2.0 * aff14 * zsd * aff15 * aff13
                - 3.0 * app14 * ajj15 * app16 * aff13
                + 2.0 * aff05 * aff10 * ass16
                - 2.0 * aff14 * zsd * aff15
                + 3.0 * app14 * ajj15 * app16
            )
            # s2 (mmpp)
            s2 = (
                -(6.0 * aff05 * aff10 * bss29 * abb9**4) / abb8**4
                - 2.0 * aff14 * zsd * aff15 * bss29 * bmm34 * bmm35
                + 2.0 * aff05 * aff10 * bss26 * bmm34 * bmm35
                + 8.0 * app08 * abb3 * app10 * bss25 * bmm34 * bmm35
                + 8.0 * aff05 * aff10 * bss29 * ass28 * aff17
                + aff14 * zsd * aff15 * bss26 * ass28 * aff17
                - 4.0 * aff14 * zsd * aff15 * bss25 * ass28 * aff17
                + 6.0 * bjj26 * ajj15 * app16 * bss25 * ass28 * aff17
                + 2.0 * abb4 * abb6 * bss21 * ass28 * aff17
                - 2.0 * bjj08 * att19 * bjj11 * ass28 * aff17
                - 4.0 * aff05 * aff10 * bss29 * aff17
                + aff14 * zsd * aff15 * bss26 * aff17
                - 4.0 * aff14 * zsd * aff15 * bss25 * aff17
                + 6.0 * bjj26 * ajj15 * app16 * bss25 * aff17
                + 2.0 * abb4 * abb6 * bss21 * aff17
                - 2.0 * bjj08 * att19 * bjj11 * aff17
                + 4.0 * aff14 * zsd * aff15 * bss29 * abb8 * abb9
                - 4.0 * aff05 * aff10 * bss26 * abb8 * abb9
                - 16.0 * app08 * abb3 * app10 * bss25 * abb8 * abb9
                - bss33 * abb8 * abb9
                + 2.0 * aff14 * zsd * aff15 * bss29 * ass23 * abb9
                - 2.0 * aff05 * aff10 * bss26 * ass23 * abb9
                - 8.0 * app08 * abb3 * app10 * bss25 * ass23 * abb9
                + bss33 * ass23 * abb9
                - 4.0 * aff05 * aff10 * bss29 * aff13
                + aff14 * zsd * aff15 * bss26 * aff13
                - 4.0 * aff14 * zsd * aff15 * bss25 * aff13
                + 6.0 * bjj26 * ajj15 * app16 * bss25 * aff13
                + 2.0 * abb4 * abb6 * bss21 * aff13
                - 2.0 * bjj08 * att19 * bjj11 * aff13
                - 2.0 * aff05 * aff10 * bss29
                - aff14 * zsd * aff15 * bss26
                + 4.0 * aff14 * zsd * aff15 * bss25
                - 6.0 * bjj26 * ajj15 * app16 * bss25
                - 2.0 * abb4 * abb6 * bss21
                + 2.0 * bjj08 * att19 * bjj11
                - (4.0 * aqq03) / aqq05
                + (44.0 * np.exp((-4.0 * tau) - 4.0 * phi) * abb3) / aqq05**2
                - (88.0 * np.exp((-6.0 * tau) - 6.0 * phi) * att19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**6) / aqq05**4
            )
            # t2 (mttt)
            t2 = (
                -(6.0 * bjj14 * ajj15 * app10 * abb9**4) / abb8**4
                + 12.0 * app21 * abb3 * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * att19 * app16 * bkk33 * bkk34
                - 7.0 * aff05 * zsd * aff10 * aqq27 * aff17
                + 8.0 * bjj14 * ajj15 * app10 * aqq27 * aff17
                + 22.0 * app08 * ajj15 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * azz19 * bjj11 * aqq27 * aff17
                - 7.0 * aff05 * zsd * aff10 * aff17
                - 4.0 * bjj14 * ajj15 * app10 * aff17
                + 22.0 * app08 * ajj15 * app10 * aff17
                - 15.0 * bjj08 * azz19 * bjj11 * aff17
                - abb4 * abb6 * abb8 * abb9
                - 24.0 * app21 * abb3 * aff15 * abb8 * abb9
                + 13.0 * aff14 * abb3 * aff15 * abb8 * abb9
                - 27.0 * bjj26 * att19 * app16 * abb8 * abb9
                + 24.0 * bjj24 * att19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * btt24 * bjj21 * abb8 * abb9
                + abb4 * abb6 * agg17 * abb9
                - 12.0 * app21 * abb3 * aff15 * agg17 * abb9
                - 13.0 * aff14 * abb3 * aff15 * agg17 * abb9
                + 27.0 * bjj26 * att19 * app16 * agg17 * abb9
                + 12.0 * bjj24 * att19 * app16 * agg17 * abb9
                - 15.0 * bjj18 * btt24 * bjj21 * agg17 * abb9
                - 7.0 * aff05 * zsd * aff10 * aff13
                - 4.0 * bjj14 * ajj15 * app10 * aff13
                + 22.0 * app08 * ajj15 * app10 * aff13
                - 15.0 * bjj08 * azz19 * bjj11 * aff13
                + 7.0 * aff05 * zsd * aff10
                - 2.0 * bjj14 * ajj15 * app10
                - 22.0 * app08 * ajj15 * app10
                + 15.0 * bjj08 * azz19 * bjj11
                - (8.0 * aqq03 * zsd) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aqq05**4
            )
            # u2 (mtte)
            u2 = (
                -(6.0 * app21 * abb3 * aff15 * abb9**4) / abb8**4
                + 6.0 * aff05 * zsd * aff10 * bkk33 * bkk34
                - 6.0 * bll16 * ajj15 * app10 * bkk33 * bkk34
                - abb4 * abb6 * aqq27 * aff17
                + 8.0 * app21 * abb3 * aff15 * aqq27 * aff17
                + 4.0 * aff14 * abb3 * aff15 * aqq27 * aff17
                - 3.0 * app14 * att19 * app16 * aqq27 * aff17
                - abb4 * abb6 * aff17
                - 4.0 * app21 * abb3 * aff15 * aff17
                + 4.0 * aff14 * abb3 * aff15 * aff17
                - 3.0 * app14 * att19 * app16 * aff17
                - 12.0 * aff05 * zsd * aff10 * abb8 * abb9
                + 12.0 * bll16 * ajj15 * app10 * abb8 * abb9
                - 6.0 * aff05 * zsd * aff10 * agg17 * abb9
                + 6.0 * bll16 * ajj15 * app10 * agg17 * abb9
                - abb4 * abb6 * aff13
                - 4.0 * app21 * abb3 * aff15 * aff13
                + 4.0 * aff14 * abb3 * aff15 * aff13
                - 3.0 * app14 * att19 * app16 * aff13
                + abb4 * abb6
                - 2.0 * app21 * abb3 * aff15
                - 4.0 * aff14 * abb3 * aff15
                + 3.0 * app14 * att19 * app16
            )
            # v2 (mttp)
            v2 = (
                (6.0 * app21 * abb3 * aff15 * avv19 * abb9**4) / abb8**4
                - 6.0 * aff05 * zsd * aff10 * avv19 * bmm34 * bmm35
                + 6.0 * app08 * ajj15 * app10 * avv19 * bmm34 * bmm35
                - 6.0 * bjj24 * att19 * app16 * bmm34 * bmm35
                + abb4 * abb6 * avv19 * ass28 * aff17
                - 8.0 * app21 * abb3 * aff15 * avv19 * ass28 * aff17
                - 4.0 * aff14 * abb3 * aff15 * avv19 * ass28 * aff17
                + 3.0 * bjj26 * att19 * app16 * avv19 * ass28 * aff17
                + 12.0 * app08 * ajj15 * app10 * ass28 * aff17
                - 12.0 * bjj08 * azz19 * bjj11 * ass28 * aff17
                + abb4 * abb6 * avv19 * aff17
                + 4.0 * app21 * abb3 * aff15 * avv19 * aff17
                - 4.0 * aff14 * abb3 * aff15 * avv19 * aff17
                + 3.0 * bjj26 * att19 * app16 * avv19 * aff17
                + 12.0 * app08 * ajj15 * app10 * aff17
                - 12.0 * bjj08 * azz19 * bjj11 * aff17
                + 12.0 * aff05 * zsd * aff10 * avv19 * abb8 * abb9
                - 12.0 * app08 * ajj15 * app10 * avv19 * abb8 * abb9
                + 9.0 * aff14 * abb3 * aff15 * abb8 * abb9
                - 24.0 * bjj26 * att19 * app16 * abb8 * abb9
                + 12.0 * bjj24 * att19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * btt24 * bjj21 * abb8 * abb9
                + 6.0 * aff05 * zsd * aff10 * avv19 * ass23 * abb9
                - 6.0 * app08 * ajj15 * app10 * avv19 * ass23 * abb9
                - 9.0 * aff14 * abb3 * aff15 * ass23 * abb9
                + 24.0 * bjj26 * att19 * app16 * ass23 * abb9
                + 6.0 * bjj24 * att19 * app16 * ass23 * abb9
                - 15.0 * bjj18 * btt24 * bjj21 * ass23 * abb9
                + abb4 * abb6 * avv19 * aff13
                + 4.0 * app21 * abb3 * aff15 * avv19 * aff13
                - 4.0 * aff14 * abb3 * aff15 * avv19 * aff13
                + 3.0 * bjj26 * att19 * app16 * avv19 * aff13
                + 12.0 * app08 * ajj15 * app10 * aff13
                - 12.0 * bjj08 * azz19 * bjj11 * aff13
                - abb4 * abb6 * avv19
                + 2.0 * app21 * abb3 * aff15 * avv19
                + 4.0 * aff14 * abb3 * aff15 * avv19
                - 3.0 * bjj26 * att19 * app16 * avv19
                - 12.0 * app08 * ajj15 * app10
                + 12.0 * bjj08 * azz19 * bjj11
                - (8.0 * aqq03 * zsd) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aqq05**4
            )
            # w2 (mtte... wait, this is mtee)
            w2 = (
                -(6.0 * aff05 * zsd * aff10 * abb9**4) / abb8**4
                + 2.0 * abb4 * abb6 * bkk33 * bkk34
                - 2.0 * aff14 * abb3 * aff15 * bkk33 * bkk34
                + (8.0 * aff05 * zsd * aff10 * aff17) / aff13
                - 4.0 * aff05 * zsd * aff10 * aff17
                - 4.0 * abb4 * abb6 * abb8 * abb9
                + 4.0 * aff14 * abb3 * aff15 * abb8 * abb9
                - 2.0 * abb4 * abb6 * agg17 * abb9
                + 2.0 * aff14 * abb3 * aff15 * agg17 * abb9
                - 4.0 * aff05 * zsd * aff10 * aff13
                - 2.0 * aff05 * zsd * aff10
            )
            # x2 (mtep)
            x2 = (
                (6.0 * aff05 * zsd * aff10 * avv19 * abb9**4) / abb8**4
                - 2.0 * abb4 * abb6 * avv19 * bmm34 * bmm35
                + 2.0 * aff14 * abb3 * aff15 * avv19 * bmm34 * bmm35
                - 4.0 * bll16 * ajj15 * app10 * bmm34 * bmm35
                - 8.0 * aff05 * zsd * aff10 * avv19 * ass28 * aff17
                + 3.0 * aff14 * abb3 * aff15 * ass28 * aff17
                - 3.0 * app14 * att19 * app16 * ass28 * aff17
                + 4.0 * aff05 * zsd * aff10 * avv19 * aff17
                + 3.0 * aff14 * abb3 * aff15 * aff17
                - 3.0 * app14 * att19 * app16 * aff17
                + 4.0 * abb4 * abb6 * avv19 * abb8 * abb9
                - 4.0 * aff14 * abb3 * aff15 * avv19 * abb8 * abb9
                + 8.0 * bll16 * ajj15 * app10 * abb8 * abb9
                + 2.0 * abb4 * abb6 * avv19 * ass23 * abb9
                - 2.0 * aff14 * abb3 * aff15 * avv19 * ass23 * abb9
                + 4.0 * bll16 * ajj15 * app10 * ass23 * abb9
                + 4.0 * aff05 * zsd * aff10 * avv19 * aff13
                + 3.0 * aff14 * abb3 * aff15 * aff13
                - 3.0 * app14 * att19 * app16 * aff13
                + 2.0 * aff05 * zsd * aff10 * avv19
                - 3.0 * aff14 * abb3 * aff15
                + 3.0 * app14 * att19 * app16
            )
            # y2 (mtpp)
            y2 = (
                -(6.0 * aff05 * zsd * aff10 * bss29 * abb9**4) / abb8**4
                + 2.0 * abb4 * abb6 * bss29 * bmm34 * bmm35
                - 2.0 * aff14 * abb3 * aff15 * bss29 * bmm34 * bmm35
                + 2.0 * aff05 * zsd * aff10 * bss26 * bmm34 * bmm35
                + 8.0 * app08 * ajj15 * app10 * bss25 * bmm34 * bmm35
                + 8.0 * aff05 * zsd * aff10 * bss29 * ass28 * aff17
                - abb4 * abb6 * bss26 * ass28 * aff17
                + aff14 * abb3 * aff15 * bss26 * ass28 * aff17
                - 6.0 * aff14 * abb3 * aff15 * bss25 * ass28 * aff17
                + 6.0 * bjj26 * att19 * app16 * bss25 * ass28 * aff17
                + abb4 * abb6 * byy24 * ass28 * aff17
                + abb4 * zsd * abb6 * bss21 * ass28 * aff17
                - 2.0 * bjj08 * azz19 * bjj11 * ass28 * aff17
                - 4.0 * aff05 * zsd * aff10 * bss29 * aff17
                - abb4 * abb6 * bss26 * aff17
                + aff14 * abb3 * aff15 * bss26 * aff17
                - 6.0 * aff14 * abb3 * aff15 * bss25 * aff17
                + 6.0 * bjj26 * att19 * app16 * bss25 * aff17
                + abb4 * abb6 * byy24 * aff17
                + abb4 * zsd * abb6 * bss21 * aff17
                - 2.0 * bjj08 * azz19 * bjj11 * aff17
                - 4.0 * abb4 * abb6 * bss29 * abb8 * abb9
                + 4.0 * aff14 * abb3 * aff15 * bss29 * abb8 * abb9
                - 4.0 * aff05 * zsd * aff10 * bss26 * abb8 * abb9
                - 16.0 * app08 * ajj15 * app10 * bss25 * abb8 * abb9
                - byy35 * abb8 * abb9
                - 2.0 * abb4 * abb6 * bss29 * ass23 * abb9
                + 2.0 * aff14 * abb3 * aff15 * bss29 * ass23 * abb9
                - 2.0 * aff05 * zsd * aff10 * bss26 * ass23 * abb9
                - 8.0 * app08 * ajj15 * app10 * bss25 * ass23 * abb9
                + byy35 * ass23 * abb9
                - 4.0 * aff05 * zsd * aff10 * bss29 * aff13
                - abb4 * abb6 * bss26 * aff13
                + aff14 * abb3 * aff15 * bss26 * aff13
                - 6.0 * aff14 * abb3 * aff15 * bss25 * aff13
                + 6.0 * bjj26 * att19 * app16 * bss25 * aff13
                + abb4 * abb6 * byy24 * aff13
                + abb4 * zsd * abb6 * bss21 * aff13
                - 2.0 * bjj08 * azz19 * bjj11 * aff13
                - 2.0 * aff05 * zsd * aff10 * bss29
                + abb4 * abb6 * bss26
                - aff14 * abb3 * aff15 * bss26
                + 6.0 * aff14 * abb3 * aff15 * bss25
                - 6.0 * bjj26 * att19 * app16 * bss25
                - abb4 * abb6 * byy24
                - abb4 * zsd * abb6 * bss21
                + 2.0 * bjj08 * azz19 * bjj11
                - (8.0 * aqq03 * zsd) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aqq05**4
            )
            # z2 (meee)
            z2 = (
                -(6.0 * abb4 * abb6 * abb9**4) / abb8**4
                + (8.0 * abb4 * abb6 * bzz9) / bzz7
                - 4.0 * abb4 * abb6 * bzz9
                - 4.0 * abb4 * abb6 * bzz7
                - 2.0 * abb4 * abb6
            )
            # a3 (meep)
            a3 = (
                (6.0 * abb4 * abb6 * aii12 * abb9**4) / abb8**4
                - (2.0 * aff14 * abb3 * aii17 * abb9**3) / abb8**3
                - (8.0 * abb4 * abb6 * aii12 * aff17) / aff13
                + 4.0 * abb4 * abb6 * aii12 * aff17
                + 4.0 * aff14 * abb3 * aii17 * abb8 * abb9
                + (2.0 * aff14 * abb3 * aii17 * abb9) / abb8
                + 4.0 * abb4 * abb6 * aii12 * aff13
                + 2.0 * abb4 * abb6 * aii12
            )
            # b3 (mepp)
            b3 = (
                -(6.0 * abb4 * abb6 * ayy24 * abb9**4) / abb8**4
                + 2.0 * abb4 * abb6 * cbb24 * bmm34 * bmm35
                + 4.0 * aff14 * abb3 * aii17 * ayy16 * bmm34 * bmm35
                + 8.0 * abb4 * abb6 * ayy24 * ass28 * aff17
                + cbb18 * ass28 * aff17
                - 4.0 * abb4 * abb6 * ayy24 * aff17
                + cbb18 * aff17
                - 4.0 * abb4 * abb6 * cbb24 * abb8 * abb9
                - 8.0 * aff14 * abb3 * aii17 * ayy16 * abb8 * abb9
                - 2.0 * abb4 * abb6 * cbb24 * ass23 * abb9
                - 4.0 * aff14 * abb3 * aii17 * ayy16 * ass23 * abb9
                - 4.0 * abb4 * abb6 * ayy24 * aff13
                + cbb18 * aff13
                - 2.0 * abb4 * abb6 * ayy24
                - 2.0 * aff14 * abb3 * aii17
                + 3.0 * app14 * att19 * cbb09
            )
            # c3 (mppp)
            c3 = (
                (6.0 * abb4 * abb6 * ccc24 * abb9**4) / abb8**4
                - 6.0 * aff14 * abb3 * aii17 * ayy24 * bmm34 * bmm35
                - 6.0 * abb4 * abb6 * ayy16 * ayy17 * bmm34 * bmm35
                - 8.0 * abb4 * abb6 * ccc24 * ass28 * aff17
                + abb4 * abb6 * ccc23 * ass28 * aff17
                + 3.0 * aff14 * abb3 * aii17 * ayy17 * ass28 * aff17
                - 3.0 * cbb18 * ayy16 * ass28 * aff17
                + 4.0 * abb4 * abb6 * ccc24 * aff17
                + abb4 * abb6 * ccc23 * aff17
                + 3.0 * aff14 * abb3 * aii17 * ayy17 * aff17
                - 3.0 * cbb18 * ayy16 * aff17
                + 12.0 * aff14 * abb3 * aii17 * ayy24 * abb8 * abb9
                + 12.0 * abb4 * abb6 * ayy16 * ayy17 * abb8 * abb9
                - ccc28 * abb8 * abb9
                + 6.0 * aff14 * abb3 * aii17 * ayy24 * ass23 * abb9
                + 6.0 * abb4 * abb6 * ayy16 * ayy17 * ass23 * abb9
                + ccc28 * ass23 * abb9
                + 4.0 * abb4 * abb6 * ccc24 * aff13
                + abb4 * abb6 * ccc23 * aff13
                + 3.0 * aff14 * abb3 * aii17 * ayy17 * aff13
                - 3.0 * cbb18 * ayy16 * aff13
                + 2.0 * abb4 * abb6 * ccc24
                - abb4 * abb6 * ccc23
                - 3.0 * aff14 * abb3 * aii17 * ayy17
                + 3.0 * cbb18 * ayy16
                - (8.0 * abb1 * zsd) / aff04
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * ajj15) / aff04**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * azz19) / aff04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**7) / aff04**4
            )
            # d3 (tttt)
            d3 = (
                -(6.0 * bjj14 * att19 * app10 * abb9**4) / abb8**4
                + 12.0 * app21 * ajj15 * aff15 * bkk33 * bkk34
                - 12.0 * bjj24 * azz19 * app16 * bkk33 * bkk34
                - 7.0 * aff05 * abb3 * aff10 * aqq27 * aff17
                + 8.0 * bjj14 * att19 * app10 * aqq27 * aff17
                + 22.0 * app08 * att19 * app10 * aqq27 * aff17
                - 15.0 * bjj08 * btt24 * bjj11 * aqq27 * aff17
                - 7.0 * aff05 * abb3 * aff10 * aff17
                - 4.0 * bjj14 * att19 * app10 * aff17
                + 22.0 * app08 * att19 * app10 * aff17
                - 15.0 * bjj08 * btt24 * bjj11 * aff17
                - abb4 * zsd * abb6 * abb8 * abb9
                - 24.0 * app21 * ajj15 * aff15 * abb8 * abb9
                + 13.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 27.0 * bjj26 * azz19 * app16 * abb8 * abb9
                + 24.0 * bjj24 * azz19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * cdd24 * bjj21 * abb8 * abb9
                + abb4 * zsd * abb6 * agg17 * abb9
                - 12.0 * app21 * ajj15 * aff15 * agg17 * abb9
                - 13.0 * aff14 * ajj15 * aff15 * agg17 * abb9
                + 27.0 * bjj26 * azz19 * app16 * agg17 * abb9
                + 12.0 * bjj24 * azz19 * app16 * agg17 * abb9
                - 15.0 * bjj18 * cdd24 * bjj21 * agg17 * abb9
                - 7.0 * aff05 * abb3 * aff10 * aff13
                - 4.0 * bjj14 * att19 * app10 * aff13
                + 22.0 * app08 * att19 * app10 * aff13
                - 15.0 * bjj08 * btt24 * bjj11 * aff13
                + 7.0 * aff05 * abb3 * aff10
                - 2.0 * bjj14 * att19 * app10
                - 22.0 * app08 * att19 * app10
                + 15.0 * bjj08 * btt24 * bjj11
                - (8.0 * aqq03 * abb3) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * att19) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * btt24) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aqq05**4
            )
            # e3 (ttte)
            e3 = (
                -(6.0 * app21 * ajj15 * aff15 * abb9**4) / abb8**4
                + 6.0 * aff05 * abb3 * aff10 * bkk33 * bkk34
                - 6.0 * bll16 * att19 * app10 * bkk33 * bkk34
                - abb4 * zsd * abb6 * aqq27 * aff17
                + 8.0 * app21 * ajj15 * aff15 * aqq27 * aff17
                + 4.0 * aff14 * ajj15 * aff15 * aqq27 * aff17
                - 3.0 * app14 * azz19 * app16 * aqq27 * aff17
                - abb4 * zsd * abb6 * aff17
                - 4.0 * app21 * ajj15 * aff15 * aff17
                + 4.0 * aff14 * ajj15 * aff15 * aff17
                - 3.0 * app14 * azz19 * app16 * aff17
                - 12.0 * aff05 * abb3 * aff10 * abb8 * abb9
                + 12.0 * bll16 * att19 * app10 * abb8 * abb9
                - 6.0 * aff05 * abb3 * aff10 * agg17 * abb9
                + 6.0 * bll16 * att19 * app10 * agg17 * abb9
                - abb4 * zsd * abb6 * aff13
                - 4.0 * app21 * ajj15 * aff15 * aff13
                + 4.0 * aff14 * ajj15 * aff15 * aff13
                - 3.0 * app14 * azz19 * app16 * aff13
                + abb4 * zsd * abb6
                - 2.0 * app21 * ajj15 * aff15
                - 4.0 * aff14 * ajj15 * aff15
                + 3.0 * app14 * azz19 * app16
            )
            # f3 (tttp)
            f3 = (
                (6.0 * app21 * ajj15 * aff15 * avv19 * abb9**4) / abb8**4
                - 6.0 * aff05 * abb3 * aff10 * avv19 * bmm34 * bmm35
                + 6.0 * app08 * att19 * app10 * avv19 * bmm34 * bmm35
                - 6.0 * bjj24 * azz19 * app16 * bmm34 * bmm35
                + abb4 * zsd * abb6 * avv19 * ass28 * aff17
                - 8.0 * app21 * ajj15 * aff15 * avv19 * ass28 * aff17
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * ass28 * aff17
                + 3.0 * bjj26 * azz19 * app16 * avv19 * ass28 * aff17
                + 12.0 * app08 * att19 * app10 * ass28 * aff17
                - 12.0 * bjj08 * btt24 * bjj11 * ass28 * aff17
                + abb4 * zsd * abb6 * avv19 * aff17
                + 4.0 * app21 * ajj15 * aff15 * avv19 * aff17
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * aff17
                + 3.0 * bjj26 * azz19 * app16 * avv19 * aff17
                + 12.0 * app08 * att19 * app10 * aff17
                - 12.0 * bjj08 * btt24 * bjj11 * aff17
                + 12.0 * aff05 * abb3 * aff10 * avv19 * abb8 * abb9
                - 12.0 * app08 * att19 * app10 * avv19 * abb8 * abb9
                + 9.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 24.0 * bjj26 * azz19 * app16 * abb8 * abb9
                + 12.0 * bjj24 * azz19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * cdd24 * bjj21 * abb8 * abb9
                + 6.0 * aff05 * abb3 * aff10 * avv19 * ass23 * abb9
                - 6.0 * app08 * att19 * app10 * avv19 * ass23 * abb9
                - 9.0 * aff14 * ajj15 * aff15 * ass23 * abb9
                + 24.0 * bjj26 * azz19 * app16 * ass23 * abb9
                + 6.0 * bjj24 * azz19 * app16 * ass23 * abb9
                - 15.0 * bjj18 * cdd24 * bjj21 * ass23 * abb9
                + abb4 * zsd * abb6 * avv19 * aff13
                + 4.0 * app21 * ajj15 * aff15 * avv19 * aff13
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * aff13
                + 3.0 * bjj26 * azz19 * app16 * avv19 * aff13
                + 12.0 * app08 * att19 * app10 * aff13
                - 12.0 * bjj08 * btt24 * bjj11 * aff13
                - abb4 * zsd * abb6 * avv19
                + 2.0 * app21 * ajj15 * aff15 * avv19
                + 4.0 * aff14 * ajj15 * aff15 * avv19
                - 3.0 * bjj26 * azz19 * app16 * avv19
                - 12.0 * app08 * att19 * app10
                + 12.0 * bjj08 * btt24 * bjj11
                - (8.0 * aqq03 * abb3) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * att19) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * btt24) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aqq05**4
            )
            # g3 (ttee)
            g3 = (
                -(6.0 * aff05 * abb3 * aff10 * abb9**4) / abb8**4
                + 2.0 * abb4 * zsd * abb6 * bkk33 * bkk34
                - 2.0 * aff14 * ajj15 * aff15 * bkk33 * bkk34
                + (8.0 * aff05 * abb3 * aff10 * aff17) / aff13
                - 4.0 * aff05 * abb3 * aff10 * aff17
                - 4.0 * abb4 * zsd * abb6 * abb8 * abb9
                + 4.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 2.0 * abb4 * zsd * abb6 * agg17 * abb9
                + 2.0 * aff14 * ajj15 * aff15 * agg17 * abb9
                - 4.0 * aff05 * abb3 * aff10 * aff13
                - 2.0 * aff05 * abb3 * aff10
            )
            # h3 (ttep)
            h3 = (
                (6.0 * aff05 * abb3 * aff10 * avv19 * abb9**4) / abb8**4
                - 2.0 * abb4 * zsd * abb6 * avv19 * bmm34 * bmm35
                + 2.0 * aff14 * ajj15 * aff15 * avv19 * bmm34 * bmm35
                - 4.0 * bll16 * att19 * app10 * bmm34 * bmm35
                - 8.0 * aff05 * abb3 * aff10 * avv19 * ass28 * aff17
                + 3.0 * aff14 * ajj15 * aff15 * ass28 * aff17
                - 3.0 * app14 * azz19 * app16 * ass28 * aff17
                + 4.0 * aff05 * abb3 * aff10 * avv19 * aff17
                + 3.0 * aff14 * ajj15 * aff15 * aff17
                - 3.0 * app14 * azz19 * app16 * aff17
                + 4.0 * abb4 * zsd * abb6 * avv19 * abb8 * abb9
                - 4.0 * aff14 * ajj15 * aff15 * avv19 * abb8 * abb9
                + 8.0 * bll16 * att19 * app10 * abb8 * abb9
                + 2.0 * abb4 * zsd * abb6 * avv19 * ass23 * abb9
                - 2.0 * aff14 * ajj15 * aff15 * avv19 * ass23 * abb9
                + 4.0 * bll16 * att19 * app10 * ass23 * abb9
                + 4.0 * aff05 * abb3 * aff10 * avv19 * aff13
                + 3.0 * aff14 * ajj15 * aff15 * aff13
                - 3.0 * app14 * azz19 * app16 * aff13
                + 2.0 * aff05 * abb3 * aff10 * avv19
                - 3.0 * aff14 * ajj15 * aff15
                + 3.0 * app14 * azz19 * app16
            )
            # i3 (ttpp)
            i3 = (
                -(6.0 * aff05 * abb3 * aff10 * bss29 * abb9**4) / abb8**4
                + 2.0 * abb4 * zsd * abb6 * bss29 * bmm34 * bmm35
                - 2.0 * aff14 * ajj15 * aff15 * bss29 * bmm34 * bmm35
                + 2.0 * aff05 * abb3 * aff10 * bss26 * bmm34 * bmm35
                + 8.0 * app08 * att19 * app10 * bss25 * bmm34 * bmm35
                + 8.0 * aff05 * abb3 * aff10 * bss29 * ass28 * aff17
                - abb4 * zsd * abb6 * bss26 * ass28 * aff17
                + aff14 * ajj15 * aff15 * bss26 * ass28 * aff17
                - 6.0 * aff14 * ajj15 * aff15 * bss25 * ass28 * aff17
                + 6.0 * bjj26 * azz19 * app16 * bss25 * ass28 * aff17
                + 4.0 * app08 * att19 * app10 * ass28 * aff17
                - 8.0 * bjj08 * btt24 * bjj11 * ass28 * aff17
                - 4.0 * aff05 * abb3 * aff10 * bss29 * aff17
                - abb4 * zsd * abb6 * bss26 * aff17
                + aff14 * ajj15 * aff15 * bss26 * aff17
                - 6.0 * aff14 * ajj15 * aff15 * bss25 * aff17
                + 6.0 * bjj26 * azz19 * app16 * bss25 * aff17
                + 4.0 * app08 * att19 * app10 * aff17
                - 8.0 * bjj08 * btt24 * bjj11 * aff17
                - 4.0 * abb4 * zsd * abb6 * bss29 * abb8 * abb9
                + 4.0 * aff14 * ajj15 * aff15 * bss29 * abb8 * abb9
                - 4.0 * aff05 * abb3 * aff10 * bss26 * abb8 * abb9
                - 16.0 * app08 * att19 * app10 * bss25 * abb8 * abb9
                + 6.0 * aff14 * ajj15 * aff15 * abb8 * abb9
                - 21.0 * bjj26 * azz19 * app16 * abb8 * abb9
                + 15.0 * bjj18 * cdd24 * bjj21 * abb8 * abb9
                - 2.0 * abb4 * zsd * abb6 * bss29 * ass23 * abb9
                + 2.0 * aff14 * ajj15 * aff15 * bss29 * ass23 * abb9
                - 2.0 * aff05 * abb3 * aff10 * bss26 * ass23 * abb9
                - 8.0 * app08 * att19 * app10 * bss25 * ass23 * abb9
                - 6.0 * aff14 * ajj15 * aff15 * ass23 * abb9
                + 21.0 * bjj26 * azz19 * app16 * ass23 * abb9
                - 15.0 * bjj18 * cdd24 * bjj21 * ass23 * abb9
                - 4.0 * aff05 * abb3 * aff10 * bss29 * aff13
                - abb4 * zsd * abb6 * bss26 * aff13
                + aff14 * ajj15 * aff15 * bss26 * aff13
                - 6.0 * aff14 * ajj15 * aff15 * bss25 * aff13
                + 6.0 * bjj26 * azz19 * app16 * bss25 * aff13
                + 4.0 * app08 * att19 * app10 * aff13
                - 8.0 * bjj08 * btt24 * bjj11 * aff13
                - 2.0 * aff05 * abb3 * aff10 * bss29
                + abb4 * zsd * abb6 * bss26
                - aff14 * ajj15 * aff15 * bss26
                + 6.0 * aff14 * ajj15 * aff15 * bss25
                - 6.0 * bjj26 * azz19 * app16 * bss25
                - 4.0 * app08 * att19 * app10
                + 8.0 * bjj08 * btt24 * bjj11
                - (8.0 * aqq03 * abb3) / aqq05
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * att19) / aqq05**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * btt24) / aqq05**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aqq05**4
            )
            # j3 (teee)
            j3 = (
                -(6.0 * abb4 * zsd * abb6 * abb9**4) / abb8**4
                + (8.0 * abb4 * zsd * abb6 * bzz9) / bzz7
                - 4.0 * abb4 * zsd * abb6 * bzz9
                - 4.0 * abb4 * zsd * abb6 * bzz7
                - 2.0 * abb4 * zsd * abb6
            )
            # k3 (teep)
            k3 = (
                (6.0 * abb4 * zsd * bdd14 * bdd15 * abb9**4) / abb8**4
                - (2.0 * aff14 * ajj15 * bdd08 * abb9**3) / abb8**3
                - (8.0 * abb4 * zsd * bdd14 * bdd15 * aff17) / aff13
                + 4.0 * abb4 * zsd * bdd14 * bdd15 * aff17
                + 4.0 * aff14 * ajj15 * bdd08 * abb8 * abb9
                + (2.0 * aff14 * ajj15 * bdd08 * abb9) / abb8
                + 4.0 * abb4 * zsd * bdd14 * bdd15 * aff13
                + 2.0 * abb4 * zsd * bdd14 * bdd15
            )
            # l3 (tepp)
            l3 = (
                -(6.0 * abb4 * zsd * bdd14 * cll17 * abb9**4) / abb8**4
                + 2.0 * abb4 * zsd * bdd14 * cll24 * bmm34 * bmm35
                + 4.0 * aff14 * ajj15 * bdd08 * cll16 * bmm34 * bmm35
                + 8.0 * abb4 * zsd * bdd14 * cll17 * ass28 * aff17
                + cll18 * ass28 * aff17
                - 4.0 * abb4 * zsd * bdd14 * cll17 * aff17
                + cll18 * aff17
                - 4.0 * abb4 * zsd * bdd14 * cll24 * abb8 * abb9
                - 8.0 * aff14 * ajj15 * bdd08 * cll16 * abb8 * abb9
                - 2.0 * abb4 * zsd * bdd14 * cll24 * ass23 * abb9
                - 4.0 * aff14 * ajj15 * bdd08 * cll16 * ass23 * abb9
                - 4.0 * abb4 * zsd * bdd14 * cll17 * aff13
                + cll18 * aff13
                - 2.0 * abb4 * zsd * bdd14 * cll17
                - 2.0 * aff14 * ajj15 * bdd08
                + 3.0 * app14 * azz19 * cll08
            )
            # m3 (tppp)
            m3 = (
                (6.0 * abb4 * zsd * abb6 * ccc24 * abb9**4) / abb8**4
                - 6.0 * aff14 * ajj15 * aii17 * ayy24 * bmm34 * bmm35
                - 6.0 * abb4 * zsd * abb6 * ayy16 * ayy17 * bmm34 * bmm35
                - 8.0 * abb4 * zsd * abb6 * ccc24 * ass28 * aff17
                + abb4 * zsd * abb6 * cmm23 * ass28 * aff17
                + 3.0 * aff14 * ajj15 * aii17 * ayy17 * ass28 * aff17
                - 3.0 * cmm16 * ayy16 * ass28 * aff17
                + 4.0 * abb4 * zsd * abb6 * ccc24 * aff17
                + abb4 * zsd * abb6 * cmm23 * aff17
                + 3.0 * aff14 * ajj15 * aii17 * ayy17 * aff17
                - 3.0 * cmm16 * ayy16 * aff17
                + 12.0 * aff14 * ajj15 * aii17 * ayy24 * abb8 * abb9
                + 12.0 * abb4 * zsd * abb6 * ayy16 * ayy17 * abb8 * abb9
                - cmm28 * abb8 * abb9
                + 6.0 * aff14 * ajj15 * aii17 * ayy24 * ass23 * abb9
                + 6.0 * abb4 * zsd * abb6 * ayy16 * ayy17 * ass23 * abb9
                + cmm28 * ass23 * abb9
                + 4.0 * abb4 * zsd * abb6 * ccc24 * aff13
                + abb4 * zsd * abb6 * cmm23 * aff13
                + 3.0 * aff14 * ajj15 * aii17 * ayy17 * aff13
                - 3.0 * cmm16 * ayy16 * aff13
                + 2.0 * abb4 * zsd * abb6 * ccc24
                - abb4 * zsd * abb6 * cmm23
                - 3.0 * aff14 * ajj15 * aii17 * ayy17
                + 3.0 * cmm16 * ayy16
                - (8.0 * abb1 * abb3) / aff04
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd**4) / aff04**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * zsd**6) / aff04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aff04**4
            )
            # n3 (eeee)
            n3 = (
                -(6.0 * abb9**4) / abb8**4
                + (8.0 * cnn5) / cnn3
                - 4.0 * cnn5
                - 4.0 * cnn3
                - 2.0
            )
            # o3 (eeep)
            o3 = (
                (6.0 * bgg4 * abb9**4) / abb8**4
                - (8.0 * bgg4 * coo9) / coo7
                + 4.0 * bgg4 * coo9
                + 4.0 * bgg4 * coo7
                + 2.0 * bgg4
            )
            # p3 (eepp)
            p3 = (
                -(6.0 * cpp08 * abb9**4) / abb8**4
                + (2.0 * cpp12 * abb9**3) / abb8**3
                + (8.0 * cpp08 * aff17) / aff13
                - 4.0 * cpp08 * aff17
                - 4.0 * cpp12 * abb8 * abb9
                - (2.0 * cpp12 * abb9) / abb8
                - 4.0 * cpp08 * aff13
                - 2.0 * cpp08
            )
            # q3 (eppp)
            q3 = (
                (6.0 * cqq20 * abb9**4) / abb8**4
                - (6.0 * cqq19 * cqq25 * abb9**3) / abb8**3
                - 8.0 * cqq20 * cqq28 * aff17
                + cqq21 * cqq28 * aff17
                + 4.0 * cqq20 * aff17
                + cqq21 * aff17
                + 12.0 * cqq19 * cqq25 * abb8 * abb9
                + (6.0 * cqq19 * cqq25 * abb9) / abb8
                + 4.0 * cqq20 * aff13
                + cqq21 * aff13
                + 2.0 * cqq20
                - ann05 * ann06
                + abb4 * zsd * bdd14
                + cqq12
                + 3.0 * app14 * azz19 * cll08
            )
            # r3 (pppp)
            r3 = (
                -(6.0 * crr19 * abb9**4) / abb8**4
                + (12.0 * crr28 * bii15 * abb9**3) / abb8**3
                - 3.0 * crr21 * ass28 * aff17
                + 8.0 * crr19 * ass28 * aff17
                - 4.0 * bii11 * crr18 * ass28 * aff17
                - 3.0 * crr21 * aff17
                - 4.0 * crr19 * aff17
                - 4.0 * bii11 * crr18 * aff17
                - 24.0 * crr28 * bii15 * abb8 * abb9
                - crr25 * abb8 * abb9
                - 12.0 * crr28 * bii15 * ass23 * abb9
                + crr25 * ass23 * abb9
                - 3.0 * crr21 * aff13
                - 4.0 * crr19 * aff13
                - 4.0 * bii11 * crr18 * aff13
                + 3.0 * crr21
                - 2.0 * crr19
                + 4.0 * bii11 * crr18
                - (8.0 * abb1 * abb3) / aff04
                + (56.0 * np.exp((-4.0 * tau) - 4.0 * phi) * zsd**4) / aff04**2
                - (96.0 * np.exp((-6.0 * tau) - 6.0 * phi) * zsd**6) / aff04**3
                + (48.0 * np.exp((-8.0 * tau) - 8.0 * phi) * zsd**8) / aff04**4
            )

            L4 = np.column_stack(
                [
                    j2,
                    k2,
                    l2,
                    m2,
                    n2,
                    o2,
                    p2,
                    q2,
                    r2,
                    s2,
                    t2,
                    u2,
                    v2,
                    w2,
                    x2,
                    y2,
                    z2,
                    a3,
                    b3,
                    c3,
                    d3,
                    e3,
                    f3,
                    g3,
                    h3,
                    i3,
                    j3,
                    k3,
                    l3,
                    m3,
                    n3,
                    o3,
                    p3,
                    q3,
                    r3,
                ]
            )
            G4 = np.column_stack(
                [
                    self.linfo[0].d4link(mu),
                    self.linfo[1].d4link(tau),
                    self.linfo[2].d4link(eps),
                    self.linfo[3].d4link(phi),
                ]
            )

        I2 = self.tri["i2"]
        I3 = self.tri["i3"]
        I4 = self.tri["i4"]

        de = gamlss_etamu(L1, L2, L3, L4, IG1, G2, G3, G4, I2, I3, I4, deriv - 1)
        ret = gamlss_gH(
            X,
            jj,
            de["l1"],
            de["l2"],
            I2,
            l3=de["l3"],
            i3=I3,
            l4=de["l4"],
            i4=I4,
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

    def residuals(
        self, y: np.ndarray, fitted: np.ndarray, rtype: str = "deviance"
    ) -> np.ndarray:
        """Residuals for shashlss.  Mirrors ``mgcv`` ``shash$residuals``."""
        y = np.asarray(y, dtype=np.float64).ravel()
        fitted = np.asarray(fitted, dtype=np.float64)
        if fitted.ndim != 2:
            raise ValueError(
                "shashlss residuals expect fitted values with shape (n, 4)."
            )
        if fitted.shape[1] != 4:
            raise ValueError("shashlss residuals expect fitted values with 4 columns.")
        if y.size != fitted.shape[0]:
            raise ValueError("y and fitted must have the same number of observations.")

        mu = np.asarray(fitted[:, 0], dtype=np.float64)
        tau = np.asarray(fitted[:, 1], dtype=np.float64)
        eps = np.asarray(fitted[:, 2], dtype=np.float64)
        phi = np.asarray(fitted[:, 3], dtype=np.float64)

        sig = np.exp(tau)
        delta = np.exp(phi)
        delinv = np.asarray(
            1.0 / np.maximum(delta, np.finfo(np.float64).eps), dtype=np.float64
        )

        # mgcv::shash$residuals uses R's besselK(x, nu), where x=0.25 and
        # nu depends on delta. scipy.special.kv reverses that order to kv(v, z).
        rsd = (
            y
            - mu
            - sig
            * delta
            * np.exp(0.25)
            * (kv((delinv + 1.0) / 2.0, 0.25) + kv((delinv - 1.0) / 2.0, 0.25))
            / np.sqrt(8.0 * np.pi)
        )

        if rtype == "response":
            return rsd
        if rtype != "deviance":
            raise ValueError("`rtype` must be 'deviance' or 'response' for shashlss")

        z = (y - mu) / np.maximum(sig * delta, np.finfo(np.float64).eps)
        d_tas_me = delta * np.arcsinh(z) - eps
        cc = np.cosh(d_tas_me)
        loglik = (
            -tau
            - 0.5 * np.log(2.0 * np.pi)
            + np.log(np.maximum(cc, np.finfo(np.float64).eps))
            - 0.5 * np.log1p(z**2)
            - 0.5 * np.sinh(d_tas_me) ** 2
        )
        return np.sign(rsd) * np.sqrt(np.maximum(0.0, -2.0 * loglik))

    def initialize(
        self,
        y: np.ndarray,
        X: np.ndarray,
        jj: list,
        offset=None,
        weights=None,
        E=None,
    ) -> np.ndarray:
        """Starting values for shash.  Mirrors mgcv shash$initialize."""
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        p = X.shape[1]
        start = np.zeros(p, dtype=np.float64)
        use_unscaled = bool(E is not None and getattr(E, "use_unscaled", False))

        # 1) Location: regress y on X1
        X1 = X[:, jj[0]]
        if E is not None and E.shape[1] > 0:
            E1 = E[:, jj[0]]
            if use_unscaled:
                start1 = _qr_coef_pivoted(
                    np.vstack([X1, E1]),
                    np.concatenate([y, np.zeros(E1.shape[0], dtype=np.float64)]),
                )
            else:
                start1 = _pen_reg(X1, E1, y)
        else:
            start1 = _qr_coef_pivoted(X1, y)
        start1 = np.where(np.isfinite(start1), start1, 0.0)
        start[jj[0]] = start1

        # 2) Log-scale: regress log|residuals| on X2
        mu_hat = X1 @ start1
        res = y - mu_hat
        log_abs_res = np.log(np.abs(res))
        X2 = X[:, jj[1]]
        if E is not None and E.shape[1] > 0:
            E2 = E[:, jj[1]]
            if use_unscaled:
                start2 = _qr_coef_pivoted(
                    np.vstack([X2, E2]),
                    np.concatenate(
                        [log_abs_res, np.zeros(E2.shape[0], dtype=np.float64)]
                    ),
                )
            else:
                start2 = _pen_reg(X2, E2, log_abs_res)
        else:
            start2 = _qr_coef_pivoted(X2, log_abs_res)
        start2 = np.where(np.isfinite(start2), start2, 0.0)
        start[jj[1]] = start2

        # 3) Skewness: initialize eps near 0 (linkfun(0) = 0 for identity)
        X3 = X[:, jj[2]]
        yt3 = np.zeros(n, dtype=np.float64)
        start3 = _qr_coef_pivoted(X3, yt3)
        start3 = np.where(np.isfinite(start3), start3, 0.0)
        start[jj[2]] = start3

        # 4) Log-kurtosis: initialize phi near 0 (linkfun(0) = 0 for identity)
        X4 = X[:, jj[3]]
        yt4 = np.zeros(n, dtype=np.float64)
        start4 = _qr_coef_pivoted(X4, yt4)
        start4 = np.where(np.isfinite(start4), start4, 0.0)
        start[jj[3]] = start4

        return start


def shashlss(b: float = 0.01, phi_pen: float = 1e-3) -> ShashlssFamily:
    """
    Sinh-arcsinh location-scale-skewness-kurtosis family.

    Parameters
    ----------
    b : float
        Lower bound parameter for the logeb link on tau. Default 0.01.
    phi_pen : float
        L2 penalty weight on the log-kurtosis parameter phi. Default 1e-3.

    Returns
    -------
    ShashlssFamily instance.
    """
    return ShashlssFamily(b=b, phi_pen=phi_pen)
