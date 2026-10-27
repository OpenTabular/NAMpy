"""Small multivariate Taylor algebra for analytic likelihood derivatives.

Coefficients are derivatives divided by multi-index factorials. Truncation
propagates exact derivatives through order four, with no finite differences
and no optional tensor/autodiff dependency in the statistical backend.
"""

from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial

import numpy as np
from scipy.special import gammaln, polygamma


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

    def expm1(self):
        return self.compose([np.expm1(self.value)] + [np.exp(self.value)] * self.order)

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
        derivatives = []
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
