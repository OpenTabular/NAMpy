"""Scalar Mathlib operations used by cancellation-sensitive R family ports."""

import math

import numpy as np


def _apply(function, values):
    """Match R src/main/arithmetic.c's per-element ``math1`` libm calls.

    One-ULP differences in exp/log feed differences of large terms in the
    third/fourth cnorm and hurdle-Poisson derivatives. Keep source arithmetic
    instead of relaxing the derivative parity checks across CPU dispatches.
    """
    array = np.asarray(values, dtype=float)

    def evaluate(value):
        try:
            return function(float(value))
        except OverflowError:
            return np.inf
        except ValueError:
            singular = value == (-1 if function is math.log1p else 0)
            return -np.inf if singular else np.nan

    return np.fromiter(
        (evaluate(value) for value in array.flat), dtype=float, count=array.size
    ).reshape(array.shape)


def exp(values):
    return _apply(math.exp, values)


def log(values):
    return _apply(math.log, values)


def expm1(values):
    return _apply(math.expm1, values)


def log1p(values):
    return _apply(math.log1p, values)


def cube(values):
    """R 4.6 arithmetic.c::R_pow for real-vector ``x^3``.

    R special-cases small bases with ordered multiplication, and otherwise
    calls scalar pow. NumPy's vector power need not use either operation.
    """
    def evaluate(value):
        if value == 0:
            return 0.0
        if -11 <= value <= 11:
            return value * value * value
        try:
            return math.pow(value, 3)
        except OverflowError:
            return math.copysign(np.inf, value)

    return _apply(evaluate, values)


def fourth(values):
    """R_pow's ordered small-base product and scalar pow for exponent four."""
    def evaluate(value):
        if value == 0:
            return 0.0
        if -11 <= value <= 11:
            return value * value * value * value
        try:
            return math.pow(value, 4)
        except OverflowError:
            return np.inf

    return _apply(evaluate, values)
