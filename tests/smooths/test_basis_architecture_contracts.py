from __future__ import annotations

import numpy as np

import nampy.gam.smooths  # noqa: F401 - attaches runtime classes
import nampy.gam.specs.smooth_build  # noqa: F401 - attaches spec builders
from nampy.gam.basis_registry import basis_descriptors, tensor_basis_names
from nampy.gam.linalg.qr import (
    r_linpack_qr_no_pivot,
    r_linpack_qty,
    r_linpack_qy,
)
from nampy.gam.smooths.registry import make_basis_term
from nampy.gam.smooths.univariate._single_penalty import (
    SinglePenaltyLowRankSmoothTerm,
)
from nampy.gam.splines._low_rank import (
    normalize_coordinate_knots,
    ordered_unique_numeric_rows,
    r_sample_without_replacement,
    top_eigensystem,
)

REGULAR_BASES = {
    "bs",
    "cc",
    "cp",
    "cr",
    "cs",
    "ds",
    "gp",
    "mrf",
    "ps",
    "sos",
    "tp",
    "ts",
}


def test_regular_basis_descriptors_own_all_construction_capabilities():
    descriptors = {
        descriptor.name: descriptor
        for descriptor in basis_descriptors()
        if descriptor.direct_runtime
    }

    assert set(descriptors) == REGULAR_BASES
    assert tensor_basis_names() == REGULAR_BASES
    for descriptor in descriptors.values():
        assert descriptor.runtime_class is not None
        assert descriptor.spec_builder is not None
        assert (
            descriptor.runtime_class.supports_tensor_marginal
            is descriptor.supports_tensor
        )


def test_generic_runtime_factory_constructs_every_regular_basis():
    for descriptor in basis_descriptors():
        if not descriptor.direct_runtime:
            continue
        if descriptor.name == "sos":
            features = ["latitude", "longitude"]
        elif descriptor.max_features is None:
            features = ["x", "z"]
        else:
            features = ["x"]

        term = make_basis_term(descriptor.name, feature=features, k=10)

        assert type(term) is descriptor.runtime_class
        assert term.basis_name == descriptor.name


def test_ds_gp_and_sos_share_the_single_penalty_runtime_lifecycle():
    descriptors = {item.name: item for item in basis_descriptors()}

    for basis in ("ds", "gp", "sos"):
        assert issubclass(
            descriptors[basis].runtime_class, SinglePenaltyLowRankSmoothTerm
        )


def test_shared_low_rank_helpers_preserve_r_sampling_and_row_order_contracts():
    np.testing.assert_array_equal(
        r_sample_without_replacement(10, 5, seed=1),
        np.array([8, 3, 6, 0, 1]),
    )
    rows = np.array([[2.0, 1.0], [1.0, 3.0], [2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_array_equal(
        ordered_unique_numeric_rows(rows),
        np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]]),
    )
    np.testing.assert_array_equal(
        normalize_coordinate_knots(([1.0, 2.0], [3.0, 4.0]), 2),
        np.array([[1.0, 3.0], [2.0, 4.0]]),
    )


def test_shared_r_linpack_qty_is_inverse_of_qy():
    rng = np.random.default_rng(42)
    packed_qr, qraux = r_linpack_qr_no_pivot(rng.normal(size=(8, 5)))
    values = rng.normal(size=(8, 3))

    np.testing.assert_allclose(
        r_linpack_qy(packed_qr, qraux, r_linpack_qty(packed_qr, qraux, values)),
        values,
        rtol=1e-14,
        atol=1e-14,
    )


def test_shared_top_eigensystem_returns_dominant_invariant_subspace():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(12, 12))
    matrix = 0.5 * (raw + raw.T)

    values, vectors = top_eigensystem(matrix, 5)
    expected = np.linalg.eigvalsh(matrix)
    expected = expected[np.argsort(np.abs(expected))[::-1]][:5]

    np.testing.assert_allclose(
        np.sort(values), np.sort(expected), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        matrix @ vectors,
        vectors * values[None, :],
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(5), atol=1e-10)
