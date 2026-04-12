from __future__ import annotations

from nampy.gam.families import GammaIdentityFamily, make_gam_family


def test_gamma_identity_family_registered():
    family = make_gam_family({"name": "gamma", "link": "identity"})
    assert isinstance(family, GammaIdentityFamily)
