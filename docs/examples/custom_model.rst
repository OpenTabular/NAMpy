Custom Model Examples
======================

Custom neural architectures inherit :class:`nampy.neural.modules.BaseModel` and
return a dictionary containing an ``"output"`` tensor. A thin sklearn wrapper
then selects the task contract.

.. code-block:: python

   from nampy.models import NeuralRegressor

   class MyRegressor(NeuralRegressor):
       def __init__(self, **kwargs):
           super().__init__(
               model=MyCustomModel,
               config=MyModelConfig,
               **kwargs,
           )

The complete architecture and configuration example is in
:doc:`../user_guide/custom_models`.
