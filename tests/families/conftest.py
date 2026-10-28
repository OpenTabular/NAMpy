"""Exercise general-family parity under valid penalty eigensystem choices."""

import pytest


@pytest.fixture(params=["default", "sign_flip", "divide_and_conquer"])
def general_penalty_orientation(request, monkeypatch):
    from nampy.gam.fit.solvers.general_family import fixed_smoothing

    if request.param == "sign_flip":
        original = fixed_smoothing._sl_single_penalty_block

        def flipped(*args, **kwargs):
            block = original(*args, **kwargs)
            if block.D.ndim == 2:
                block.D[:, 2] *= -1
                block.Di[2, :] *= -1
            return block

        monkeypatch.setattr(fixed_smoothing, "_sl_single_penalty_block", flipped)
    elif request.param == "divide_and_conquer":
        original = fixed_smoothing.scipy_eigh

        def alternate(*args, **kwargs):
            kwargs["driver"] = "evd"
            return original(*args, **kwargs)

        monkeypatch.setattr(fixed_smoothing, "scipy_eigh", alternate)
