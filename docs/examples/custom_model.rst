Custom Model Examples
======================

Custom neural architectures inherit :class:`nampy.neural.architectures.BaseModel`
and return a dictionary containing an ``"output"`` tensor. Registering that
architecture generates its regression, classification, and LSS estimators.

.. code-block:: python

   from nampy.models import estimator_family
   from nampy.neural.registry import NeuralArchitecture, register_architecture

   register_architecture(NeuralArchitecture(
       name="my_model",
       estimator_prefix="MyModel",
       module_path=f"{__name__}:MyCustomModel",
       config_path=f"{__name__}:MyModelConfig",
       capabilities=frozenset({
           "regression", "classification", "distributional"
       }),
   ))
   estimators = estimator_family("my_model", module_name=__name__)
   MyRegressor = estimators.regressor
   MyClassifier = estimators.classifier
   MyLSS = estimators.lss

The complete architecture and configuration example is in
:doc:`../user_guide/custom_models`.
