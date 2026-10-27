"""Runnable reference example: neural_reference/extensions/NeuralEstimatorFamily."""

from nampy.models import estimator_family, NeuralEstimatorFamily

registered = estimator_family('nam', module_name=__name__)
result = NeuralEstimatorFamily(regressor=registered.regressor,
                              classifier=registered.classifier, lss=registered.lss)
assert result.regressor is not None
print(result)
