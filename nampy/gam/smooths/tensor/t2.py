"""SS-ANOVA-style alternative tensor-product smooth (``t2``)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ...constraints.absorption import apply_linear_constraint
from ...penalties import penalty_id_for_local_index, rescale_tensor_penalties_for_fit
from ...splines.basis.natparam import nat_param_type3
from ..algebra import rowwise_kronecker
from ..registry import register_smooth
from ..smooth_base import BaseSmoothTerm, _normalize_knots, build_penalty_definition
from .marginals import (
    build_tensor_marginal_terms,
    resolve_tensor_marginal_features,
    tensor_marginal_fit_matrices,
    tensor_marginal_predict_matrix,
    validate_tensor_marginal_bases,
)


@dataclass
class T2Block:
    matrix: np.ndarray
    label: str
    order: int
    penalized: bool
    column_labels: tuple[str, ...] = ()


@dataclass
class T2MatrixResult:
    matrix: np.ndarray
    penalty_widths: tuple[int, ...]
    penalty_labels: tuple[str, ...]
    block_orders: tuple[int, ...]
    null_space_dim: int


def _column_products(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return rowwise_kronecker([left, right])


def _normalize_t2_ord(ord_value, n_marginals: int):
    if ord_value is None:
        return None
    values = np.asarray(
        [ord_value] if np.isscalar(ord_value) else ord_value,
        dtype=np.float64,
    ).ravel()
    values = np.rint(values).astype(int)
    if not np.any((values >= 0) & (values <= int(n_marginals))):
        warnings.warn("ord is wrong. reset to None.", stacklevel=3)
        return None
    if np.any((values < 0) | (values > int(n_marginals))):
        warnings.warn(
            "ord contains out of range orders (which will be ignored)",
            stacklevel=3,
        )
    return tuple(int(value) for value in values)


def t2_model_matrix(marginal_bases, ranks, *, full=True, ord=None) -> T2MatrixResult:
    """Port of ``mgcv:::t2.model.matrix`` including its block ordering."""
    bases = [np.asarray(value, dtype=np.float64) for value in marginal_bases]
    ranks = [int(value) for value in ranks]
    if len(bases) == 0 or len(bases) != len(ranks):
        raise ValueError("t2 requires equally sized non-empty bases and ranks lists.")
    n = bases[0].shape[0]
    if any(B.ndim != 2 or B.shape[0] != n for B in bases):
        raise ValueError(
            "All t2 marginal bases must be matrices with equal row counts."
        )
    if any(rank < 0 or rank > B.shape[1] for B, rank in zip(bases, ranks, strict=True)):
        raise ValueError("A t2 marginal penalty rank is outside its basis dimension.")

    first = bases[0]
    first_rank = ranks[0]
    blocks = [
        T2Block(
            matrix=first[:, :first_rank],
            label="r",
            order=1,
            penalized=True,
        )
    ]
    no_null = first_rank >= first.shape[1]
    if not no_null:
        null = first[:, first_rank:]
        labels = tuple(str(i) for i in range(1, null.shape[1] + 1))
        blocks.append(
            T2Block(
                matrix=null,
                label="n",
                order=0,
                penalized=False,
                column_labels=labels,
            )
        )

    for basis, rank in zip(bases[1:], ranks[1:], strict=True):
        previous = list(blocks)
        range_basis = basis[:, :rank]
        null_exists = rank < basis.shape[1]
        if not null_exists:
            no_null = True
        null_basis = basis[:, rank:] if null_exists else None
        null_labels = (
            tuple(str(i) for i in range(1, null_basis.shape[1] + 1))
            if null_exists
            else ()
        )

        next_blocks: list[T2Block] = []
        for block in previous:
            if not full or block.penalized:
                next_blocks.append(
                    T2Block(
                        matrix=_column_products(block.matrix, range_basis),
                        label=f"{block.label}r",
                        order=block.order + 1,
                        penalized=True,
                    )
                )
            else:
                labels = block.column_labels or tuple(
                    str(i) for i in range(1, block.matrix.shape[1] + 1)
                )
                for column, label in enumerate(labels):
                    next_blocks.append(
                        T2Block(
                            matrix=block.matrix[:, [column]] * range_basis,
                            label=f"{label}r",
                            order=block.order + 1,
                            penalized=True,
                        )
                    )

        if null_exists:
            for block in previous:
                if not full or not block.penalized:
                    product = _column_products(block.matrix, null_basis)
                    if full:
                        left_labels = block.column_labels or tuple(
                            str(i) for i in range(1, block.matrix.shape[1] + 1)
                        )
                        product_labels = tuple(
                            f"{left}{right}"
                            for left in left_labels
                            for right in null_labels
                        )
                    else:
                        product_labels = ()
                    next_blocks.append(
                        T2Block(
                            matrix=product,
                            label=f"{block.label}n",
                            order=block.order,
                            penalized=False if full else block.penalized,
                            column_labels=product_labels,
                        )
                    )
                else:
                    for column, label in enumerate(null_labels):
                        next_blocks.append(
                            T2Block(
                                matrix=block.matrix * null_basis[:, [column]],
                                label=f"{block.label}{label}",
                                order=block.order,
                                penalized=True,
                            )
                        )
        blocks = next_blocks

    if ord is not None:
        allowed = {int(value) for value in ord}
        blocks = [block for block in blocks if block.order in allowed]
        if 0 not in allowed:
            no_null = True
    if len(blocks) == 0:
        return T2MatrixResult(
            matrix=np.zeros((n, 0), dtype=np.float64),
            penalty_widths=(),
            penalty_labels=(),
            block_orders=(),
            null_space_dim=0,
        )

    matrix = np.hstack([block.matrix for block in blocks])
    penalized_blocks = blocks if no_null else blocks[:-1]
    widths = tuple(int(block.matrix.shape[1]) for block in penalized_blocks)
    labels = tuple(str(block.label) for block in penalized_blocks)
    null_dim = 0 if no_null else int(blocks[-1].matrix.shape[1])
    return T2MatrixResult(
        matrix=np.asarray(matrix, dtype=np.float64),
        penalty_widths=widths,
        penalty_labels=labels,
        block_orders=tuple(int(block.order) for block in blocks),
        null_space_dim=null_dim,
    )


def t2_identity_penalties(n_coef: int, widths) -> list[np.ndarray]:
    penalties = []
    start = 0
    for width in widths:
        stop = start + int(width)
        diagonal = np.zeros(int(n_coef), dtype=np.float64)
        diagonal[start:stop] = 1.0
        penalties.append(np.diag(diagonal))
        start = stop
    return penalties


@register_smooth("t2")
class AlternativeTensorProductSplineTerm(BaseSmoothTerm):
    term_type = "tensor_t2"
    basis_name = "t2"
    supports_tensor_marginal = False

    def __init__(
        self,
        feature,
        k=5,
        basis="cr",
        m=None,
        xt=None,
        label=None,
        term_id=None,
        smoothing_id=None,
        by=None,
        sp=None,
        select=False,
        fixed=False,
        null_penalty_tol=1e-10,
        knots=None,
        pc=None,
        full=False,
        ord=None,
        metadata=None,
    ):
        del fixed
        features = list(feature) if not isinstance(feature, (str, int)) else [feature]
        if len(features) < 1:
            raise ValueError("AlternativeTensorProductSplineTerm requires a feature.")
        super().__init__(
            feature=features,
            label=label or f"t2({', '.join(map(str, features))})",
            term_id=term_id,
            smoothing_id=smoothing_id,
            by=by,
            sp=sp,
            metadata=metadata,
        )
        self.k = [int(k)] * len(features) if np.isscalar(k) else [int(v) for v in k]
        if len(self.k) != len(features):
            raise ValueError(f"k must have length {len(features)}, got {self.k}.")
        self.basis = (
            [str(basis)] * len(features)
            if isinstance(basis, str)
            else [str(value) for value in basis]
        )
        if len(self.basis) != len(features):
            raise ValueError(
                f"basis must have length {len(features)}, got {self.basis}."
            )
        self.basis = validate_tensor_marginal_bases(self.basis)
        self.m = m
        self.xt = xt
        self.select = bool(select)
        self.fixed = False
        self.null_penalty_tol = float(null_penalty_tol)
        self.knots = _normalize_knots(knots, features)
        self.pc = pc
        self.full = bool(full)
        self.ord = _normalize_t2_ord(ord, len(features))

        self._feature_indices = None
        self._feature_names = None
        self._marginals = None
        self._marginal_transforms = None
        self._marginal_ranks = None
        self._basis_train = None
        self._penalties = None
        self._raw_basis_train = None
        self._raw_penalties = None
        self._penalty_labels = None
        self._penalty_orders = None
        self._null_space_dim = None
        self._by_state = None

    @property
    def expected_linked_penalty_count(self):
        return None

    def _reparameterized_marginals(self, X):
        setup_bases = []
        local_bases = []
        transforms = []
        ranks = []
        for marginal in self._marginals:
            setup_basis, penalty, _ = tensor_marginal_fit_matrices(
                marginal,
                centered=False,
                apply_np=False,
                x_train=None,
            )
            linked_setup = getattr(marginal, "shared_basis_setup", None)
            linked = (
                isinstance(linked_setup, dict)
                and str(linked_setup.get("mode", "")).lower() == "linked_id"
                and bool(linked_setup.get("pooled_feature_values"))
            )
            local_basis = (
                tensor_marginal_predict_matrix(marginal, X, centered=False)
                if linked
                else setup_basis
            )
            parameterization = nat_param_type3(setup_basis, penalty, unit_fnorm=True)
            transform = np.asarray(parameterization["P"], dtype=np.float64)
            setup_bases.append(np.asarray(parameterization["X"], dtype=np.float64))
            local_bases.append(np.asarray(local_basis @ transform, dtype=np.float64))
            transforms.append(transform)
            ranks.append(int(parameterization["rank"]))
        return setup_bases, local_bases, transforms, ranks

    def fit(self, X, feature_names):
        self._set_by_state(X, feature_names)
        shared_setups = self._linked_id_marginal_setups(self.feature)
        marginals, _, _ = build_tensor_marginal_terms(
            feature=self.feature,
            k=self.k,
            basis=self.basis,
            m=self.m,
            xt=self.xt,
            knots=self.knots,
            centered=False,
            shared_basis_setups=shared_setups,
            metadata=self.metadata,
        )
        for marginal in marginals:
            marginal.fit(X, feature_names)
            if str(getattr(marginal, "basis_name", "")).lower() == "mrf":
                self.metadata = dict(marginal.metadata)
        self._marginals = marginals
        feature_indices, feature_names_resolved = resolve_tensor_marginal_features(
            marginals
        )
        setup_bases, local_bases, transforms, ranks = self._reparameterized_marginals(X)
        setup_result = t2_model_matrix(setup_bases, ranks, full=self.full, ord=self.ord)
        local_result = t2_model_matrix(local_bases, ranks, full=self.full, ord=self.ord)
        if setup_result.penalty_widths != local_result.penalty_widths:
            raise RuntimeError("Linked t2 setup and local block structures differ.")

        raw_penalties = t2_identity_penalties(
            setup_result.matrix.shape[1], setup_result.penalty_widths
        )
        scaled_penalties, scales = rescale_tensor_penalties_for_fit(
            setup_result.matrix,
            raw_penalties,
            return_scales=True,
        )

        if self.sp is not None:
            n_sp = int(np.asarray([self.sp] if np.isscalar(self.sp) else self.sp).size)
            if n_sp != len(scaled_penalties):
                warnings.warn("length of sp incorrect in t2: ignored", stacklevel=2)
                self.sp = None

        constraint_kind = None
        constraint_transform = None
        factor_by_meta = (
            isinstance(self.metadata, dict)
            and self.metadata.get("factor_by", None) is not None
        )
        if self.pc is not None:
            max_index = max(feature_indices)

            def point_basis_fn(point):
                point_data = np.zeros((point.shape[0], max_index + 1), dtype=np.float64)
                point_data[:, feature_indices] = point
                marginal_values = [
                    tensor_marginal_predict_matrix(marginal, point_data, centered=False)
                    @ transform
                    for marginal, transform in zip(marginals, transforms, strict=True)
                ]
                return t2_model_matrix(
                    marginal_values, ranks, full=self.full, ord=self.ord
                ).matrix

            basis, penalties, constraint_transform, _ = self._apply_point_constraint(
                local_result.matrix,
                scaled_penalties,
                self.pc,
                feature_names=feature_names_resolved,
                point_basis_fn=point_basis_fn,
                fixed=False,
            )
            constraint_kind = "pc"
        elif local_result.null_space_dim > 0 and (
            self._by_state.is_constant or factor_by_meta
        ):
            n_penalized = int(sum(local_result.penalty_widths))
            constraint_row = np.zeros(local_result.matrix.shape[1], dtype=np.float64)
            if local_result.null_space_dim == 1:
                constraint_row[-1] = 1.0
            else:
                constraint_row[n_penalized:] = np.sum(
                    setup_result.matrix[:, n_penalized:], axis=0
                )
            basis, penalties, constraint_transform = apply_linear_constraint(
                local_result.matrix,
                scaled_penalties,
                constraint_row,
            )
            basis = self._apply_cached_by(basis)
            constraint_kind = "sum_to_zero"
        else:
            basis = self._apply_cached_by(local_result.matrix)
            penalties = scaled_penalties

        self._feature_indices = feature_indices
        self._feature_names = feature_names_resolved
        self._set_resolved_features(feature_names_resolved)
        self._marginal_transforms = transforms
        self._marginal_ranks = ranks
        self._raw_basis_train = np.asarray(local_result.matrix, dtype=np.float64)
        self._raw_penalties = [
            np.asarray(value, dtype=np.float64) for value in raw_penalties
        ]
        self._basis_train = np.asarray(basis, dtype=np.float64)
        self._penalties = [np.asarray(value, dtype=np.float64) for value in penalties]
        self._penalty_labels = tuple(local_result.penalty_labels)
        self._penalty_orders = tuple(local_result.block_orders[: len(penalties)])
        self._null_space_dim = int(local_result.null_space_dim)
        self._set_penalty_rescale_factors(scales)
        self._record_constraint_result(
            constraint_kind,
            constraint_transform,
            absorbed_by=("runtime" if constraint_transform is not None else None),
        )
        self.basis_name = "t2(" + ",".join(self.basis) + ")"
        return self

    def get_penalty_definitions(self):
        self._require_fitted()
        raw = list(self.penalties)
        sp_values = self._normalized_term_sp(len(raw))
        definitions = []
        for index, penalty in enumerate(raw):
            sid = (
                None
                if self.smoothing_id is None
                else penalty_id_for_local_index(
                    self.smoothing_id, index, n_penalties=len(raw)
                )
            )
            sp_value = sp_values[index] if index < len(sp_values) else None
            definitions.append(
                build_penalty_definition(
                    self,
                    penalty,
                    kind="smooth",
                    smoothing_id=sid,
                    sp_value_in=sp_value,
                    metadata_extra={
                        "term_sp": sp_value,
                        "is_selection_penalty": False,
                        "t2_component": self._penalty_labels[index],
                        "t2_order": self._penalty_orders[index],
                        "t2_full": self.full,
                    },
                    local_penalty_index=index,
                )
            )
        definitions.extend(
            self._build_selection_penalty_definitions(
                raw,
                null_penalty_tol=self.null_penalty_tol,
            )
        )
        return definitions

    def transform_new(self, X_new):
        self._require_fitted()
        marginal_values = [
            tensor_marginal_predict_matrix(marginal, X_new, centered=False) @ transform
            for marginal, transform in zip(
                self._marginals, self._marginal_transforms, strict=True
            )
        ]
        raw = t2_model_matrix(
            marginal_values,
            self._marginal_ranks,
            full=self.full,
            ord=self.ord,
        ).matrix
        return self._apply_constraint_transform_and_by(raw, X_new)


__all__ = [
    "AlternativeTensorProductSplineTerm",
    "T2MatrixResult",
    "t2_identity_penalties",
    "t2_model_matrix",
]
