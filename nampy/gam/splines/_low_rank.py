"""Shared low-rank spline setup and R-compatible sampling primitives."""

from __future__ import annotations

import math

import numpy as np


def normalize_coordinate_knots(knots, dimension: int):
    """Collect per-coordinate knot vectors for a multivariate smooth."""
    if knots is None:
        return None

    dimension = int(dimension)
    if isinstance(knots, (list, tuple)):
        if dimension == 1 and (not knots or np.isscalar(knots[0])):
            return np.asarray(knots, dtype=np.float64).reshape(-1, 1)
        if len(knots) != dimension or any(value is None for value in knots):
            return None
        columns = [np.asarray(value, dtype=np.float64).ravel() for value in knots]
        if any(column.size != columns[0].size for column in columns[1:]):
            raise ValueError(
                "components of knots relating to a single smooth must be of same length"
            )
        return np.asarray(np.column_stack(columns), dtype=np.float64)

    values = np.asarray(knots, dtype=np.float64)
    if values.ndim == 1:
        if dimension != 1:
            return None
        return values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != dimension:
        return None
    return np.asarray(values, dtype=np.float64)


def ordered_unique_numeric_rows(values):
    """Mirror ``uniquecombs(x, ordered=TRUE)`` for numeric matrices."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] == 1:
        return np.unique(matrix[:, 0]).reshape(-1, 1)

    first_by_key = {}
    for row in matrix:
        key = "*".join(format(float(value), ".15g") for value in row)
        first_by_key.setdefault(key, np.asarray(row, dtype=np.float64).copy())
    return np.vstack([first_by_key[key] for key in sorted(first_by_key)])


class RMersenneTwister:
    """Minimal R-compatible MT19937 stream used by ``temp.seed``/``sample``."""

    def __init__(self, seed):
        state = int(seed) & 0xFFFFFFFF
        for _ in range(50):
            state = (69069 * state + 1) & 0xFFFFFFFF
        state = (69069 * state + 1) & 0xFFFFFFFF
        self._state = []
        for _ in range(624):
            state = (69069 * state + 1) & 0xFFFFFFFF
            self._state.append(state)
        self._index = 624

    def _twist(self):
        for index in range(624):
            word = (self._state[index] & 0x80000000) | (
                self._state[(index + 1) % 624] & 0x7FFFFFFF
            )
            self._state[index] = (
                self._state[(index + 397) % 624]
                ^ (word >> 1)
                ^ (0x9908B0DF if word & 1 else 0)
            )
        self._index = 0

    def uniform(self):
        if self._index >= 624:
            self._twist()
        word = self._state[self._index]
        self._index += 1
        word ^= word >> 11
        word ^= (word << 7) & 0x9D2C5680
        word ^= (word << 15) & 0xEFC60000
        word ^= word >> 18
        return float(word & 0xFFFFFFFF) / float(2**32)

    def uniform_index(self, size):
        size = int(size)
        bits = int(math.ceil(math.log2(size)))
        mask = (1 << bits) - 1
        while True:
            value = 0
            for _ in range(0, bits + 1, 16):
                value = 65536 * value + math.floor(self.uniform() * 65536)
            value &= mask
            if value < size:
                return int(value)


def r_sample_without_replacement(size, sample_size, seed):
    """Mirror modern R ``sample.int(..., replace=FALSE)`` rejection sampling."""
    available = list(range(int(size)))
    remaining = int(size)
    rng = RMersenneTwister(seed)
    selected = []
    for _ in range(int(sample_size)):
        index = rng.uniform_index(remaining)
        selected.append(available[index])
        remaining -= 1
        available[index] = available[remaining]
    return np.asarray(selected, dtype=np.intp)


def low_rank_setup_locations(values, shift, knots, *, max_knots, seed):
    values = np.asarray(values, dtype=np.float64)
    if knots is not None:
        return np.asarray(knots, dtype=np.float64), False
    unique = ordered_unique_numeric_rows(values) - np.asarray(
        shift, dtype=np.float64
    )[None, :]
    if values.shape[0] > int(max_knots) and unique.shape[0] > int(max_knots):
        indices = r_sample_without_replacement(
            unique.shape[0], int(max_knots), int(seed)
        )
        return np.asarray(unique[indices, :], dtype=np.float64), True
    return np.asarray(unique, dtype=np.float64), False


def parse_low_rank_xt(xt, *, basis_name: str):
    """Parse the shared ``max.knots``/``seed`` low-rank setup options."""
    max_knots = 2000
    seed = 1
    if xt is None:
        return max_knots, seed
    if not isinstance(xt, dict):
        raise NotImplementedError(
            f"For bs={basis_name!r}, xt must be None or a dict with optional keys "
            "{'max.knots', 'seed'}."
        )
    if xt.get("max.knots") is not None:
        max_knots = int(xt["max.knots"])
    if xt.get("seed") is not None:
        seed = int(xt["seed"])
    if max_knots < 1:
        raise ValueError(
            f"For bs={basis_name!r}, xt['max.knots'] must be positive."
        )
    return max_knots, seed


def top_eigensystem(matrix, k, *, tolerance_exponent=0.7):
    """Return mgcv/Rlanczos-compatible dominant eigenpairs."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    k = int(k)

    if k <= 0 or n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros((n, 0), dtype=np.float64)
    if k > n:
        raise ValueError(f"k must be <= matrix dimension, got k={k}, n={n}.")

    tolerance = float(np.finfo(np.float64).eps ** float(tolerance_exponent))
    check_frequency = min(max(10, k // 2), max(1, n // 10))

    random_state = 1
    start = np.empty(n, dtype=np.float64)
    for index in range(n):
        random_state = (random_state * 106 + 1283) % 6075
        start[index] = float(random_state) / 6075.0 - 0.5
    start /= np.linalg.norm(start)
    vectors = [start]

    diagonal = np.zeros(n, dtype=np.float64)
    off_diagonal = np.zeros(n, dtype=np.float64)
    eigenvalues = None
    eigenvectors = None
    final_size = n
    positive_keep = k
    negative_keep = 0

    for step in range(n):
        residual = matrix @ vectors[step]
        diagonal[step] = float(vectors[step] @ residual)
        if step == 0:
            residual = residual - diagonal[step] * vectors[step]
        else:
            residual = (
                residual
                - diagonal[step] * vectors[step]
                - off_diagonal[step - 1] * vectors[step - 1]
            )
            for index in range(step + 1):
                residual += -float(residual @ vectors[index]) * vectors[index]
            for index in range(step + 1):
                residual += -float(residual @ vectors[index]) * vectors[index]

        off_diagonal[step] = float(np.linalg.norm(residual))
        if step < n - 1:
            if off_diagonal[step] == 0.0:
                raise np.linalg.LinAlgError("Lanczos breakdown in spline eigensystem.")
            vectors.append(residual / off_diagonal[step])

        should_check = (
            (step >= k and step % check_frequency == 0) or step == n - 1
        )
        if not should_check:
            continue

        tridiagonal = np.diag(diagonal[: step + 1].copy())
        if step > 0:
            off = off_diagonal[:step].copy()
            tridiagonal += np.diag(off, 1) + np.diag(off, -1)
        ascending, ascending_vectors = np.linalg.eigh(tridiagonal)
        eigenvalues = np.asarray(ascending[::-1], dtype=np.float64)
        eigenvectors = np.asarray(ascending_vectors[:, ::-1], dtype=np.float64)
        errors = np.abs(off_diagonal[step] * eigenvectors[-1, :])

        if step < k:
            continue
        max_error = max(abs(eigenvalues[0]), abs(eigenvalues[step])) * tolerance
        positive_index = 0
        negative_index = 0
        converged = True
        while positive_index + negative_index < k:
            if abs(eigenvalues[positive_index]) >= abs(
                eigenvalues[step - negative_index]
            ):
                if errors[positive_index] > max_error:
                    converged = False
                    break
                positive_index += 1
            else:
                if errors[negative_index] > max_error:
                    converged = False
                    break
                negative_index += 1
        if converged:
            positive_keep = positive_index
            negative_keep = negative_index
            final_size = step + 1
            break

    if eigenvalues is None or eigenvectors is None:
        raise np.linalg.LinAlgError("Failed to compute spline eigensystem.")

    result_vectors = np.zeros((n, k), dtype=np.float64)
    for column in range(positive_keep):
        coefficients = eigenvectors[:final_size, column]
        for index in range(final_size):
            result_vectors[:, column] += vectors[index] * coefficients[index]

    for column in range(positive_keep, positive_keep + negative_keep):
        source = final_size - (negative_keep + positive_keep - column)
        coefficients = eigenvectors[:final_size, source]
        for index in range(final_size):
            result_vectors[:, column] += vectors[index] * coefficients[index]

    result_values = np.zeros(k, dtype=np.float64)
    result_values[:positive_keep] = eigenvalues[:positive_keep]
    for column in range(positive_keep, positive_keep + negative_keep):
        source = final_size - (negative_keep + positive_keep - column)
        result_values[column] = eigenvalues[source]
    return result_values, result_vectors


__all__ = [
    "RMersenneTwister",
    "low_rank_setup_locations",
    "normalize_coordinate_knots",
    "ordered_unique_numeric_rows",
    "parse_low_rank_xt",
    "r_sample_without_replacement",
    "top_eigensystem",
]
