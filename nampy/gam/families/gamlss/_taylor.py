"""Small multivariate Taylor algebra for analytic likelihood derivatives.

Coefficients are derivatives divided by multi-index factorials. Truncation
propagates exact derivatives through order four, with no finite differences
and no optional tensor/autodiff dependency in the statistical backend.
"""

from functools import lru_cache
from itertools import combinations_with_replacement
from math import comb, factorial

import numpy as np
from scipy.special import erfcx, expit, gammaln, log_ndtr, polygamma


def _normal_mills_tail_derivatives(z, depth=64):
    """Derivatives 0..3 of m(z)-z for z>=8, by Laplace's continued fraction.

    DLMF 7.9.1, rescaled to the standard normal, gives
    m(z)-z = 1/(z + 2/(z + 3/(z + ...))). Differentiate the rational
    convergents analytically, without subtracting nearly equal powers of z.
    The 64-level evaluation is checked against 128 levels and high-precision
    log-erfc derivatives, including the branch boundary and far positive tail.
    """
    tail = [np.zeros_like(z) for _ in range(4)]
    for numerator in range(depth, 0, -1):
        denominator = [z + tail[0], 1 + tail[1], tail[2], tail[3]]
        current = [numerator / denominator[0]]
        for degree in range(1, 4):
            current.append(-sum(
                comb(degree, j) * denominator[j] * current[degree-j]
                for j in range(1, degree+1)
            ) / denominator[0])
        tail = current
    return tail


@lru_cache(None)
def indices(dimension, order):
    result = [(0,) * dimension]
    for degree in range(1, order + 1):
        for combination in combinations_with_replacement(range(dimension), degree):
            result.append(tuple(combination.count(k) for k in range(dimension)))
    return tuple(result)


class Taylor:
    __array_priority__ = 1000

    def __init__(self, value, dimension, order, variable=None):
        self.dimension, self.order = dimension, order
        self.coefficients = {(0,) * dimension: np.asarray(value, dtype=float)}
        if variable is not None and order:
            index = tuple(int(k == variable) for k in range(dimension))
            self.coefficients[index] = np.ones_like(value, dtype=float)

    @property
    def value(self):
        return self.coefficients[(0,) * self.dimension]

    def constant(self, value):
        return Taylor(value, self.dimension, self.order)

    def __add__(self, other):
        other = other if isinstance(other, Taylor) else self.constant(other)
        self._check_compatible(other)
        result = self.constant(0)
        result.coefficients = self.coefficients.copy()
        for index, value in other.coefficients.items():
            result.coefficients[index] = result.coefficients.get(index, 0) + value
        return result

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Taylor) else self.constant(other)
        self._check_compatible(other)
        result = self.constant(0)
        for a, av in self.coefficients.items():
            for b, bv in other.coefficients.items():
                index = tuple(x + y for x, y in zip(a, b, strict=True))
                if sum(index) <= self.order:
                    result.coefficients[index] = (
                        result.coefficients.get(index, 0) + av * bv
                    )
        return result

    __rmul__ = __mul__

    def _check_compatible(self, other):
        if (self.dimension, self.order) != (other.dimension, other.order):
            raise ValueError("Taylor operands must have equal dimensions and orders.")

    def subset(self, mask):
        result = self.constant(self.value[mask])
        result.coefficients = {
            key: np.broadcast_to(value, self.value.shape)[mask]
            for key, value in self.coefficients.items()
        }
        return result

    @classmethod
    def merge(cls, mask, yes, no):
        yes._check_compatible(no)
        result = yes.constant(np.zeros(mask.shape))
        result.coefficients = {}
        for key in yes.coefficients.keys() | no.coefficients.keys():
            values = np.zeros(mask.shape)
            values[mask] = yes.coefficients.get(key, 0)
            values[~mask] = no.coefficients.get(key, 0)
            result.coefficients[key] = values
        return result

    def __pow__(self, exponent):
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Taylor powers must be nonnegative integers.")
        result = self.constant(1)
        for _ in range(exponent):
            result = result * self
        return result

    def compose(self, derivatives):
        delta = self - self.value
        result, power = self.constant(derivatives[0]), self.constant(1)
        for k in range(1, self.order + 1):
            power = power * delta
            result = result + power * (derivatives[k] / factorial(k))
        return result

    def exp(self):
        value = np.exp(self.value)
        return self.compose([value] * (self.order + 1))

    def log_normal_survival(self):
        """Stable log Phi(-z), with analytic derivatives through order four."""
        if self.order > 4:
            raise NotImplementedError("Normal log-survival derivatives stop at order four.")
        z = self.value
        value = log_ndtr(-z)
        mills = np.empty_like(z)
        positive = z >= 0
        mills[positive] = np.sqrt(2/np.pi)/erfcx(z[positive]/np.sqrt(2))
        mills[~positive] = np.exp(-.5*z[~positive]**2 - .5*np.log(2*np.pi) - value[~positive])
        derivatives = [value, -mills, mills*(z-mills)]
        if self.order >= 3:
            first = mills*(mills-z)
            second = first*(2*mills-z)-mills
            third = second*(2*mills-z)+2*first*(first-1)
            tail = z >= 8
            if tail.any():
                fraction = _normal_mills_tail_derivatives(z[tail])
                second[tail], third[tail] = fraction[2], fraction[3]
            derivatives.extend([-second, -third])
        return self.compose(derivatives[:self.order+1])

    def expm1(self):
        return self.compose([np.expm1(self.value)] + [np.exp(self.value)] * self.order)

    def expm1_over_x(self):
        """Analytic expm1(x)/x, including all derivatives at zero.

        f^(k)(x)=integral_0^1 t**k exp(x*t) dt. The convergent Taylor
        series is evaluated for |x|<=4; outside use x*f'=exp(x)-f.
        Forty-eight terms bound the series remainder below float64 precision
        for orders through four on that interval.
        """
        if self.order > 4:
            raise NotImplementedError("expm1_over_x derivatives are available through order four.")
        x = self.value
        small = np.abs(x) <= 4
        derivatives: list[np.ndarray] = []
        for degree in range(self.order + 1):
            value = np.empty_like(x)
            coefficients = [1 / (factorial(j)*(j+degree+1)) for j in range(48)]
            value[small] = np.polynomial.polynomial.polyval(x[small], coefficients)
            value[~small] = (np.expm1(x[~small])/x[~small] if degree == 0 else
                             (np.exp(x[~small])-degree*derivatives[-1][~small])/x[~small])
            derivatives.append(value)
        return self.compose(derivatives)

    def softplus(self):
        """Stable log(1+exp(x)) with exact derivatives through order four."""
        p = expit(self.value)
        derivatives = [
            np.logaddexp(0, self.value), p, p*(1-p),
            p*(1-p)*(1-2*p), p*(1-p)*(1-6*p+6*p*p),
        ]
        if self.order > 4:
            raise NotImplementedError("softplus derivatives are implemented through order four.")
        return self.compose(derivatives[:self.order+1])

    def log1p(self):
        x = self.value
        return self.compose(
            [np.log1p(x)]
            + [
                (-1) ** (k - 1) * factorial(k - 1) / (1 + x) ** k
                for k in range(1, self.order + 1)
            ]
        )

    def log1p_over_x(self, *, minus_one=False):
        """Analytic log1p(x)/x, with a removable singularity at zero.

        The optional subtraction is performed inside the series so the small
        remainder is retained even when x is below machine epsilon.
        """
        x = self.value
        small = np.abs(x) <= 0.25
        derivatives: list[np.ndarray] = []
        for degree in range(self.order + 1):
            out = np.empty_like(x)
            coefficients = np.zeros(41)
            first = max(degree, int(minus_one))
            for power in range(first, 41):
                coefficients[power - degree] = (
                    (-1) ** power
                    * factorial(power)
                    / factorial(power - degree)
                    / (power + 1)
                )
            out[small] = np.polynomial.polynomial.polyval(x[small], coefficients)
            if degree == 0:
                out[~small] = np.log1p(x[~small]) / x[~small] - int(minus_one)
            else:
                previous = derivatives[-1][~small]
                if degree == 1 and minus_one:
                    previous = previous + 1
                out[~small] = (
                    (-1) ** (degree - 1)
                    * factorial(degree - 1)
                    / (1 + x[~small]) ** degree
                    - degree * previous
                ) / x[~small]
            derivatives.append(out)
        return self.compose(derivatives)

    def log(self):
        x = self.value
        derivatives = [np.log(x)]
        derivatives.extend(
            (-1) ** (k - 1) * factorial(k - 1) / x**k for k in range(1, self.order + 1)
        )
        return self.compose(derivatives)

    def lgamma(self):
        return self.compose(
            [gammaln(self.value)]
            + [polygamma(k - 1, self.value) for k in range(1, self.order + 1)]
        )

    def reciprocal(self):
        return self.compose(
            [
                (-1) ** k * factorial(k) / self.value ** (k + 1)
                for k in range(self.order + 1)
            ]
        )

    def __truediv__(self, other):
        return self * (other.reciprocal() if isinstance(other, Taylor) else 1 / other)

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    def packed_derivatives(self, degree):
        columns = []
        for index in indices(self.dimension, degree):
            if sum(index) == degree:
                multiplier = np.prod([factorial(k) for k in index])
                columns.append(
                    np.broadcast_to(self.coefficients.get(index, 0), self.value.shape)
                    * multiplier
                )
        return np.column_stack(columns)
