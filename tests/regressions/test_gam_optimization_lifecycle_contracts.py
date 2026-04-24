from __future__ import annotations

from tests._optimization_lifecycle_registry import OPTIMIZATION_LIFECYCLE_CASES


def test_optimization_lifecycle_registry_tracks_current_supported_branch_matrix():
    """Verify that lifecycle registry tracks the current supported optimizer branches."""
    assert {case.case_id for case in OPTIMIZATION_LIFECYCLE_CASES} == {
        "poisson_reml_newton_two_cr",
        "poisson_reml_bfgs_two_cr",
        "poisson_reml_efs_two_cr",
        "poisson_reml_optim_two_cr",
        "gaulss_reml_efs_two_cr",
        "gaulss_ml_newton_two_cr",
        "gamma_reml_newton_joint_scale_cr",
        "negbin_est_reml_newton_joint_theta_cr",
    }


def test_optimization_lifecycle_registry_marks_only_joint_trace_gaps():
    """Verify that lifecycle registry keeps current branch gaps explicit and narrow."""
    gap_ids = {
        case.case_id
        for case in OPTIMIZATION_LIFECYCLE_CASES
        if case.status == "known_gap"
    }
    assert gap_ids == {
        "gamma_reml_newton_joint_scale_cr",
        "negbin_est_reml_newton_joint_theta_cr",
    }


def test_optimization_lifecycle_registry_case_ids_are_unique():
    """Verify that lifecycle registry does not duplicate case identifiers."""
    ids = [case.case_id for case in OPTIMIZATION_LIFECYCLE_CASES]
    assert len(ids) == len(set(ids))
