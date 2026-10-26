"""The public soap composition recipes run without local upstream sources."""

import numpy as np

from examples.example_soap_compositions import fit_examples


def test_soap_compositions_example():
    for model, frame in fit_examples().values():
        prediction, se = model.predict(frame, return_se=True)
        assert prediction.shape == se.shape == (len(frame),)
        assert np.isfinite(prediction).all()
        assert np.isfinite(se).all()
        terms = model.predict_terms(frame)
        reconstructed = sum(value for key, value in terms.items()
                            if key not in {"output", "response"})
        np.testing.assert_allclose(reconstructed, terms["output"], atol=2e-12, rtol=2e-12)
