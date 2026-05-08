from __future__ import annotations

from tests._optimization_lifecycle_registry import OPTIMIZATION_LIFECYCLE_CASES


def test_optimization_lifecycle_registry_tracks_current_supported_branch_matrix():
    """Verify that lifecycle registry tracks the current supported optimizer branches."""
    assert {case.case_id for case in OPTIMIZATION_LIFECYCLE_CASES} == {
        "binomial_ml_efs_cr",
        "binomial_reml_bfgs_cr",
        "binomial_reml_newton_cr",
        "gamma_reml_bfgs_cr_known_gap",
        "poisson_reml_newton_two_cr",
        "poisson_reml_bfgs_two_cr",
        "poisson_reml_efs_two_cr",
        "poisson_reml_optim_two_cr",
        "poisson_reml_newton_tensor_cr",
        "poisson_reml_efs_tensor_cr",
        "gaussian_ml_newton_two_cr",
        "gaussian_reml_newton_fs_xt_ps",
        "gaussian_reml_newton_random_effect",
        "gaussian_reml_newton_re_custom_xt",
        "gaussian_reml_newton_two_cr",
        "gamma_reml_newton_joint_scale_cr",
        "negbin_est_reml_newton_joint_theta_cr",
        "negbin_fixed_theta_reml_newton_cr",
    }


def test_optimization_lifecycle_registry_marks_only_joint_trace_gaps():
    """Verify that lifecycle registry keeps current branch gaps explicit and narrow."""
    gap_ids = {
        case.case_id
        for case in OPTIMIZATION_LIFECYCLE_CASES
        if case.status == "known_gap"
    }
    assert gap_ids == {
        "gamma_reml_bfgs_cr_known_gap",
        "gamma_reml_newton_joint_scale_cr",
    }


def test_optimization_lifecycle_registry_case_ids_are_unique():
    """Verify that lifecycle registry does not duplicate case identifiers."""
    ids = [case.case_id for case in OPTIMIZATION_LIFECYCLE_CASES]
    assert len(ids) == len(set(ids))
