"""R scalar Mathlib semantics used before cancellation-sensitive derivatives."""

import math

import numpy as np
import pytest

from nampy.gam.families import _scalar_math


@pytest.mark.parametrize("name", ["exp", "log", "expm1", "log1p"])
def test_scalar_math_retains_libm_results_and_array_shape(name):
    # These intermediates feed large, nearly cancelling tail derivatives.
    values = np.array([[1e-12, 0.26236426446749106], [6.0, 6.564958885017789]])
    expected = np.array([getattr(math, name)(value) for value in values.flat])
    actual = getattr(_scalar_math, name)(values)
    assert actual.shape == values.shape
    np.testing.assert_array_equal(actual.ravel(), expected)
    assert getattr(_scalar_math, name)(values[0, 1]).shape == ()
    assert getattr(_scalar_math, name)(np.empty((2, 0))).shape == (2, 0)


@pytest.mark.parametrize(
    "name,values,expected",
    [
        ("exp", [np.nan, -np.inf, np.inf, -1000, 1000, -0.0],
         [np.nan, 0, np.inf, 0, np.inf, 1]),
        ("expm1", [np.nan, -np.inf, np.inf, -1000, 1000, -0.0],
         [np.nan, -1, np.inf, -1, np.inf, -0.0]),
        ("log", [np.nan, -np.inf, np.inf, 0, -0.0, -1],
         [np.nan, np.nan, np.inf, -np.inf, -np.inf, np.nan]),
        ("log1p", [np.nan, -np.inf, np.inf, -1, -2, -0.0],
         [np.nan, np.nan, np.inf, -np.inf, np.nan, -0.0]),
    ],
)
def test_scalar_math_preserves_source_boundary_values(name, values, expected):
    actual = getattr(_scalar_math, name)(values)
    np.testing.assert_array_equal(actual, expected)
    zero = np.asarray(expected) == 0
    np.testing.assert_array_equal(np.signbit(actual[zero]), np.signbit(np.asarray(expected)[zero]))


def test_source_cube_preserves_r_pow_branch_boundaries():
    values = [-np.inf, -1e200, -12.1, -11, -0.3, -0.0, 0, 0.3, 11, 12.1, 1e200, np.inf, np.nan]
    expected = [
        -np.inf, -np.inf, math.pow(-12.1, 3), -11 * -11 * -11,
        -0.3 * -0.3 * -0.3, 0, 0, 0.3 * 0.3 * 0.3,
        11 * 11 * 11, math.pow(12.1, 3), np.inf, np.inf, np.nan,
    ]
    np.testing.assert_array_equal(_scalar_math.cube(values), expected)
