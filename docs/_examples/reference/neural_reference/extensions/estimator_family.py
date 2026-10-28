"""Runnable reference example: neural_reference/extensions/estimator_family."""

from nampy.models import estimator_family, NeuralEstimatorFamily

registered = estimator_family('nam', module_name=__name__)
result = NeuralEstimatorFamily(regressor=registered.regressor,
                              classifier=registered.classifier, lss=registered.lss)
print(result)
