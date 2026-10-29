"""One artifact contract for neural, likelihood and fixed-loss estimators."""

import json
import pickle

import pytest

from nampy import ExpectileGAM, GAMRegressor, NAMRegressor, QuantileGAM


@pytest.mark.parametrize("cls", [NAMRegressor, GAMRegressor, ExpectileGAM, QuantileGAM])
def test_shared_artifact_preserves_estimator_type(cls, tmp_path):
    estimator = cls()
    path = estimator.save_model(tmp_path / "estimator.nampy")
    restored = cls.load_model(path)
    assert type(restored) is cls
    with path.open("rb") as handle:
        assert handle.readline() == b"NAMPY-ESTIMATOR 6\n"
        envelope = json.loads(handle.readline())
    assert envelope["format"] == "nampy-estimator"
    assert envelope["estimator_class"] == cls.__name__
    other = QuantileGAM if cls is not QuantileGAM else ExpectileGAM
    with pytest.raises(TypeError, match="contains"):
        other.load_model(path)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"format": "other", "version": 1},
        {"format": "nampy-estimator", "version": 0},
    ],
)
@pytest.mark.parametrize("cls", [NAMRegressor, GAMRegressor, ExpectileGAM, QuantileGAM])
def test_shared_artifact_rejects_unsupported_envelopes(payload, cls, tmp_path):
    path = tmp_path / "invalid.nampy"
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="not a supported NAMpy estimator artifact"):
        cls.load_model(path)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_previous_readable_artifact_headers_are_rejected_before_payload(version, tmp_path):
    path = tmp_path / "previous.nampy"
    path.write_bytes(f"NAMPY-ESTIMATOR {version}\n".encode() + b"this payload must never be unpickled")
    with pytest.raises(ValueError, match="not a supported NAMpy estimator artifact"):
        GAMRegressor.load_model(path)
