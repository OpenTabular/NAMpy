"""Identical held-out observations, honest fit eligibility and proper scores."""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats
from sklearn.base import BaseEstimator

from nampy.contracts import DistributionFitStatus
from nampy.distributions import (
    AtomMixtureLaw,
    DistributionPrediction,
    DistributionSpec,
    GammaLaw,
    NegativeBinomialLaw,
    NormalLaw,
)
from nampy.models import DistributionCandidate, compare_distributions


class _LogOnly:
    spec = replace(NormalLaw.spec, operations=frozenset({"log_prob"}))

    def validate_parameters(self, parameters, *, context):
        NormalLaw().validate_parameters(parameters, context=context)

    def evaluate(self, operation, parameters, **kwargs):
        return NormalLaw().evaluate(operation, parameters, **kwargs)


class _InfiniteLog(_LogOnly):
    def evaluate(self, operation, parameters, **kwargs):
        return np.full(len(parameters["loc"]), np.inf)


class _BrokenCalibration(_LogOnly):
    spec = NormalLaw.spec

    def evaluate(self, operation, parameters, **kwargs):
        if operation == "quantile":
            warnings.warn("diagnostic warning retained", RuntimeWarning, stacklevel=2)
            raise ValueError("quantile diagnostic failed deliberately")
        return super().evaluate(operation, parameters, **kwargs)


class _UnknownAtoms:
    base = AtomMixtureLaw(NormalLaw(), (0.0,), ("p",))
    spec = replace(base.spec, atom_locations=None)

    def validate_parameters(self, parameters, **kwargs):
        return self.base.validate_parameters(parameters, **kwargs)

    def evaluate(self, operation, parameters, **kwargs):
        return self.base.evaluate(operation, parameters, **kwargs)


class _Candidate(BaseEstimator):
    def __init__(self, loc=0.0, kind="normal", state="converged", behavior="ok"):
        self.loc, self.kind, self.state, self.behavior = loc, kind, state, behavior

    def fit(self, X, y, sample_weight=None, offset=None, context=None):
        self.X_received_ = np.asarray(X).copy()
        self.y_received_ = y.copy()
        self.weights_received_ = None if sample_weight is None else sample_weight.copy()
        self.offset_received_ = offset
        self.context_received_ = context
        if self.behavior == "raise_fit":
            warnings.warn("fit warning before failure", UserWarning, stacklevel=2)
            raise ValueError("fit failed deliberately")
        if self.behavior == "mutate":
            X[0, 0] = 999
            y[0] = 999
        if self.behavior == "warn":
            warnings.warn("fit warning retained", RuntimeWarning, stacklevel=2)
        self.fitted_ = True
        return self

    def distribution_fit_status(self):
        return DistributionFitStatus(
            "external", self.state, "explicit test backend evidence"
        )

    def predict_distribution(self, X, **kwargs):
        if self.behavior == "raise_predict":
            raise RuntimeError("prediction failed deliberately")
        n = len(X) - int(self.behavior == "omit")
        if self.kind == "gamma":
            return DistributionPrediction(
                GammaLaw(), {"shape": np.full(n, 2.0), "rate": np.ones(n)}
            )
        if self.kind == "discrete":
            return DistributionPrediction(
                NegativeBinomialLaw(), {"mean": np.ones(n), "dispersion": np.ones(n)}
            )
        if self.kind.startswith("mixed"):
            law = (
                _UnknownAtoms()
                if self.kind == "mixed_unknown"
                else AtomMixtureLaw(
                    NormalLaw(), (0.0 if self.kind == "mixed0" else 1.0,), ("p",)
                )
            )
            return DistributionPrediction(
                law, {"loc": np.zeros(n), "scale": np.ones(n), "p": np.full(n, 0.2)}
            )
        loc = np.full(n, self.loc, dtype=float)
        scale = np.ones(n)
        if self.behavior == "missing":
            loc[0] = scale[0] = np.nan
        law = (
            _InfiniteLog()
            if self.kind == "infinite"
            else _BrokenCalibration()
            if self.kind == "bad_calibration"
            else _LogOnly()
            if self.kind == "log_only"
            else NormalLaw()
        )
        return DistributionPrediction(law, {"loc": loc, "scale": scale})


class _NoStatus(_Candidate):
    distribution_fit_status = None


class _BadClone(_Candidate):
    def __sklearn_clone__(self):
        raise TypeError("cannot clone this candidate")


class _SameClone(_Candidate):
    def __sklearn_clone__(self):
        return self


def _inputs():
    return (
        np.arange(8.0).reshape(4, 2),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.arange(6.0).reshape(3, 2),
        np.array([1.0, 2.0, 3.0]),
    )


def test_common_weighted_scores_ties_copies_and_unweighted_calibration():
    X, y, V, z = _inputs()
    first = _Candidate(1.0, behavior="mutate")
    second = _Candidate(2.0, behavior="warn")
    result = compare_distributions(
        {"first": first, "second": second, "tie": _Candidate(2.0)},
        X,
        y,
        V,
        z,
        sample_weight=[1, 2, 3, 4],
        validation_weight=[1, 0, 3],
        response_units="outcome units",
    )
    assert result.ranking == ("second", "tie", "first")
    assert [row.rank for row in result.results] == [3, 1, 1]
    assert result.response_units == "outcome units"
    for row in result.results:
        assert row.eligible and row.status == "ok"
        assert row.fit_status.state == "converged"
        assert_allclose(
            row.mean_log_prob,
            np.average(stats.norm.logpdf(z, row.estimator.loc), weights=[1, 0, 3]),
        )
        assert row.n_validation == row.n_scored == 2
        assert row.validation_weight_sum == 4
        assert_allclose(row.effective_n, 1.6)
        assert row.calibration.n_used.tolist() == [2, 2, 2]
        assert row.fit_seconds >= 0 and row.evaluation_seconds >= 0
        assert_array_equal(row.estimator.X_received_, X)
        assert_array_equal(row.estimator.y_received_, y)
        assert_array_equal(row.estimator.weights_received_, [1, 2, 3, 4])
        assert not row.log_prob.flags.writeable
    assert not hasattr(first, "fitted_") and not hasattr(second, "fitted_")
    assert X[0, 0] == 0 and y[0] == 1
    assert any("fit warning retained" in value for value in result.results[1].warnings)
    frame = result.to_frame()
    assert frame.name.tolist() == ["second", "tie", "first"]
    assert (
        frame.calibration_weighting.tolist() == ["unweighted_positive_weight_rows"] * 3
    )
    assert str(frame["rank"].dtype) == "Int64"


def test_failures_nonconvergence_unknown_status_and_cloning_remain_visible():
    X, y, V, z = _inputs()
    result = compare_distributions(
        {
            "good": _Candidate(),
            "failed": _Candidate(behavior="raise_fit"),
            "nonconverged": _Candidate(state="nonconverged"),
            "unknown": _NoStatus(),
            "prediction": _Candidate(behavior="raise_predict"),
            "clone": _BadClone(),
        },
        X,
        y,
        V,
        z,
    )
    rows = {row.name: row for row in result.results}
    assert result.ranking == ("good",)
    assert (
        rows["failed"].failure_stage == "fit"
        and "fit failed deliberately" in rows["failed"].failure_reason
    )
    assert rows["clone"].failure_stage == "clone"
    assert "warning before failure" in rows["failed"].warnings[0]
    assert rows["nonconverged"].status == rows["unknown"].status == "ineligible_fit"
    assert rows["prediction"].status == "evaluation_error"
    assert rows["prediction"].fit_status.state == "converged"
    assert rows["prediction"].failure_stage == "prediction"
    assert all(row.rank is None for row in result.results if not row.eligible)
    assert "failed" in result.models and "clone" not in result.models


def test_explicit_external_status_adapter_and_optional_calibration():
    result = compare_distributions(
        {
            "custom": DistributionCandidate(
                _NoStatus(kind="log_only"),
                status_reader=lambda model: DistributionFitStatus(
                    "external", "completed", "custom training finished"
                ),
            )
        },
        *_inputs(),
    )
    row = result.results[0]
    assert row.eligible and row.fit_status.state == "completed"
    assert row.calibration is None and "requires CDF" in row.calibration_error
    assert (
        compare_distributions(
            {"good": _Candidate()}, *_inputs(), calibration_probs=None
        )
        .results[0]
        .calibration_error
        is None
    )


def test_zero_probability_is_a_valid_worst_score_and_zero_weights_do_not_multiply_infinity():
    X, y, V, z = _inputs()
    z[0] = -1
    result = compare_distributions(
        {"normal": _Candidate(), "gamma": _Candidate(kind="gamma")}, X, y, V, z
    )
    row = result.results[1]
    assert row.eligible and row.mean_log_prob == -np.inf and row.rank == 2
    assert row.n_zero_probability == 1 and row.n_scored == 3
    weighted = compare_distributions(
        {"gamma": _Candidate(kind="gamma")}, X, y, V, z, validation_weight=[0, 1, 1]
    )
    assert np.isfinite(weighted.results[0].mean_log_prob)
    assert weighted.results[0].n_zero_probability == 0


@pytest.mark.parametrize("behavior", ["missing", "omit"])
def test_candidate_specific_row_loss_is_ineligible_without_shrinking_other_scores(
    behavior,
):
    result = compare_distributions(
        {"good": _Candidate(), "bad": _Candidate(behavior=behavior)}, *_inputs()
    )
    assert result.ranking == ("good",)
    assert result.results[0].n_validation == result.results[0].n_scored == 3
    assert result.results[1].status == "evaluation_error"
    if behavior == "missing":
        result = compare_distributions(
            {"ignored": _Candidate(behavior=behavior)},
            *_inputs(),
            validation_weight=[0, 1, 1],
        )
        assert result.results[0].eligible


def test_positive_infinite_log_density_is_ineligible():
    result = compare_distributions(
        {"good": _Candidate(), "singular": _Candidate(kind="infinite")}, *_inputs()
    )
    assert result.ranking == ("good",)
    assert result.results[1].status == "evaluation_error"
    assert "NaN/+inf" in result.results[1].failure_reason


def test_diagnostic_failure_does_not_change_likelihood_eligibility_or_hide_warnings():
    result = compare_distributions(
        {"diagnostic": _Candidate(kind="bad_calibration")}, *_inputs()
    )
    row = result.results[0]
    assert row.eligible and row.rank == 1 and row.status == "ok"
    assert row.failure_stage is None
    assert row.calibration is None
    assert "quantile diagnostic failed" in row.calibration_error
    assert any("diagnostic warning retained" in message for message in row.warnings)


@pytest.mark.parametrize(
    "kinds", [["normal", "discrete"], ["normal", "mixed0"], ["mixed0", "mixed1"]]
)
def test_incompatible_measures_are_symmetrically_unranked(kinds):
    result = compare_distributions(
        {name: _Candidate(kind=name) for name in kinds}, *_inputs()
    )
    assert result.ranking == ()
    assert all(row.status == "incompatible_measure" for row in result.results)
    assert all(
        row.mean_log_prob is not None and row.rank is None for row in result.results
    )


def test_common_mixed_measure_is_rankable_and_unknown_atoms_are_explicit():
    result = compare_distributions(
        {
            "a": _Candidate(kind="mixed0"),
            "b": _Candidate(kind="mixed0"),
            "unknown": _Candidate(kind="mixed_unknown"),
        },
        *_inputs(),
    )
    assert result.ranking == ("a", "b")
    assert result.results[0].measure == ("mixed", (0.0,))
    assert result.results[2].status == "evaluation_error"
    assert "atom_locations" in result.results[2].failure_reason


@pytest.mark.parametrize(
    "override",
    [
        {"y_valid": [1, np.nan, 2]},
        {"y_train": [[1], [2], [3], [4]]},
        {"validation_weight": [0, 0, 0]},
        {"validation_weight": [1, -1, 1]},
        {"validation_weight": [1, 2]},
        {"sample_weight": [1, 2, np.inf, 4]},
        {"X_valid": np.ones((2, 2))},
        {"calibration_probs": [0.9, 0.1]},
    ],
)
def test_common_inputs_fail_before_any_candidate_fit(override):
    X, y, V, z = _inputs()
    args = dict(X_train=X, y_train=y, X_valid=V, y_valid=z, **{})
    args.update(override)
    with pytest.raises(ValueError):
        compare_distributions({"candidate": _Candidate()}, **args)


def test_candidate_argument_guards_and_invalid_status_adapters():
    with pytest.raises(ValueError, match="common"):
        DistributionCandidate(_Candidate(), fit_kwargs={"sample_weight": [1]})
    with pytest.raises(ValueError, match="common"):
        DistributionCandidate(_Candidate(), predict_kwargs={"X": [[1]]})
    with pytest.raises(ValueError):
        compare_distributions({}, *_inputs())
    result = compare_distributions(
        {"bad": DistributionCandidate(_Candidate(), status_reader=lambda model: True)},
        *_inputs(),
    )
    assert result.results[0].failure_stage == "status"
    assert "DistributionFitStatus" in result.results[0].failure_reason
    template = _SameClone()
    result = compare_distributions({"same": template}, *_inputs())
    assert result.results[0].failure_stage == "clone"
    assert not hasattr(template, "fitted_")
    assert "same" not in result.models


def test_atom_metadata_validation():
    with pytest.raises(ValueError, match="atoms"):
        replace(NormalLaw.spec, atom_locations=(0.0,))
    with pytest.raises(ValueError, match="increasing"):
        DistributionSpec(
            "bad", ("p",), frozenset(), "v1", "mixed", atom_locations=(1.0, 0.0)
        )
