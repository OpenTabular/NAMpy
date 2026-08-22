Configurations
==============

Configuration classes for nampy models.

.. currentmodule:: nampy.neural.configs

.. autosummary::
   :toctree: generated/
   :nosignatures:

   DefaultNAMConfig
   DefaultSIANConfig
   DefaultNBMConfig
   DefaultNBMSPAMConfig
   DefaultSPAMConfig
   DefaultNATTConfig
   DefaultNAMformerConfig
   DefaultLinRegConfig
   DefaultGPNAMConfig
   DefaultIGANNConfig
   DefaultQNAMConfig
   DefaultSNAMConfig
   DefaultSplineNAMConfig
   DefaultTreeNAMConfig
   DefaultEnsembleTreeNAMConfig
   DefaultNodeGAMConfig

Using Configurations
--------------------

Configurations can be passed to models or used to create custom settings:

.. code-block:: python

   from nampy.models import NAMRegressor
   from nampy.neural.configs import DefaultNAMConfig
   
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

Each architecture-specific configuration is exported from :mod:`nampy.neural.configs`.
