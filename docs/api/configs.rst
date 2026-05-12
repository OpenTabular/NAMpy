Configurations
==============

Configuration classes for nampy models.

.. currentmodule:: nampy.configs

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DefaultNAMConfig
   DefaultNBMConfig
   DefaultNATTConfig
   DefaultNAMformerConfig
   DefaultLinRegConfig
   DefaultSparseNAMConfig
   DefaultSplineNAMConfig
   DefaultNodeGAMConfig

Using Configurations
--------------------

Configurations can be passed to models or used to create custom settings:

.. code-block:: python

   from nampy.models import NAMRegressor
   from nampy.configs import DefaultNAMConfig
   
   # Use default config (implicit)
   model = NAMRegressor()
   
   # Or pass hyperparameters directly
   model = NAMRegressor(
       lr=1e-3,
       dropout=0.3,
       layer_sizes=[128, 64, 32]
   )
   
   # For custom models, use config explicitly
   config = DefaultNAMConfig()
   config.lr = 1e-3
   config.dropout = 0.3

Note
----

Some models (like GPNAM and QNAM) use ``DefaultNAMConfig`` instead of having
their own dedicated configuration classes.
