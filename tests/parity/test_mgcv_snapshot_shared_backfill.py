"""Collect the shared snapshot-parity classes that no other module re-exports.

`tests/_mgcv_snapshot_parity_shared.py` is itself uncollectable (leading
underscore), so every class defined there must be re-exported by a collected
module or it silently stops running. `test_mgcv_snapshot_parity.py` owns
`TestMgcvParity` and `test_mgcv_snapshot_extended_matrix.py` /
`test_mgcv_parity_failing_and_warnings.py` own the re-exported
`TestAdditionalScenarioParity` methods; this module owns everything else.

Keep these re-exports at module scope so pytest collects the shared classes.
"""

from tests._mgcv_snapshot_parity_shared import (
    TestAdditionalScenarioParity as _SharedTestAdditionalScenarioParity,
)
from tests._mgcv_snapshot_parity_shared import (
    TestGaussianPriorWeights as _SharedTestGaussianPriorWeights,
)
from tests._mgcv_snapshot_parity_shared import (
    TestGaussianPriorWeightsMgcvParity as _SharedTestGaussianPriorWeightsMgcvParity,
)
from tests._mgcv_snapshot_parity_shared import (
    TestMgcvDeviancePenaltyScaleAssembly as _SharedTestMgcvDeviancePenaltyScaleAssembly,
)
from tests._mgcv_snapshot_parity_shared import (
    TestNumericByVariable as _SharedTestNumericByVariable,
)
from tests._mgcv_snapshot_parity_shared import (
    TestPSplineSmooth as _SharedTestPSplineSmooth,
)
from tests._mgcv_snapshot_parity_shared import (
    TestSmoothingMethodParity as _SharedTestSmoothingMethodParity,
)


class TestMgcvDeviancePenaltyScaleAssembly(_SharedTestMgcvDeviancePenaltyScaleAssembly):
    pass


class TestGaussianPriorWeights(_SharedTestGaussianPriorWeights):
    pass


class TestGaussianPriorWeightsMgcvParity(_SharedTestGaussianPriorWeightsMgcvParity):
    pass


class TestPSplineSmooth(_SharedTestPSplineSmooth):
    pass


class TestNumericByVariable(_SharedTestNumericByVariable):
    pass


class TestSmoothingMethodParity(_SharedTestSmoothingMethodParity):
    pass


class TestAdditionalScenarioTensorBackfill:
    """The shared tensor fixed-sp scenarios no other module re-exports."""

    test_gaussian_te_ps_ps_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_ps_ps_fixed_matches_mgcv
    )
    test_gaussian_ti_ps_ps_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_ti_ps_ps_fixed_matches_mgcv
    )
    test_gaussian_te_tp_ps_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_te_tp_ps_fixed_matches_mgcv
    )
    test_gaussian_ti_tp_ps_fixed_matches_mgcv = (
        _SharedTestAdditionalScenarioParity.test_gaussian_ti_tp_ps_fixed_matches_mgcv
    )
