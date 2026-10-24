Custom Models
=============

NAMpy provides a structured way to implement custom models while leveraging
the existing infrastructure.

Overview
--------

To create a custom model, you need to:

1. Define a configuration class
2. Implement the model architecture
3. Register the architecture and generate its estimator family
4. Use your custom model

Step-by-Step Guide
------------------

1. Define Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Create a dataclass for your model's hyperparameters:

.. code-block:: python

   from dataclasses import dataclass
   
   @dataclass
   class MyModelConfig:
       lr: float = 1e-4
       lr_patience: int = 10
       weight_decay: float = 1e-6
       lr_factor: float = 0.1
       hidden_size: int = 128
       num_layers: int = 3
       dropout: float = 0.3

2. Implement the Model
~~~~~~~~~~~~~~~~~~~~~~~

Create your model by inheriting from `BaseModel`:

.. code-block:: python

   from nampy.neural.architectures import BaseModel
   import torch
   import torch.nn as nn
   
   class MyCustomModel(BaseModel):
       def __init__(
           self,
           cat_feature_info,
           num_feature_info,
           num_classes: int = 1,
           config=None,
           **kwargs,
       ):
           super().__init__(**kwargs)
           self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
           
           if config is None:
               config = MyModelConfig()
           # Calculate input size
           total_input_size = (
               sum(int(info.get("dimension", 1)) for info in num_feature_info.values())
               + sum(int(info.get("dimension", 1)) for info in cat_feature_info.values())
           )
           
           # Define your architecture
           layers = []
           in_size = total_input_size
           
           for _ in range(config.num_layers):
               layers.append(nn.Linear(in_size, config.hidden_size))
               layers.append(nn.ReLU())
               layers.append(nn.Dropout(config.dropout))
               in_size = config.hidden_size
           
           layers.append(nn.Linear(in_size, num_classes))
           
           self.mlp = nn.Sequential(*layers)
       
       def forward(self, num_features: dict, cat_features: dict) -> dict:
           """
           Forward pass of the model.
           
           Parameters
           ----------
           num_features : dict
               Dictionary of numerical features with feature names as keys.
           cat_features : dict
               Dictionary of categorical features with feature names as keys.
           
           Returns
           -------
           dict
               Dictionary containing the output tensor.
           """
           blocks = [tensor.float() for tensor in [*num_features.values(), *cat_features.values()]]
           if not blocks:
               raise ValueError("At least one feature block is required.")
           input_tensor = torch.cat(blocks, dim=1)

           # Forward pass
           output = self.mlp(input_tensor)
           
           # MUST return a dictionary with "output" key
           return {"output": output}

3. Register the Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Declare supported capabilities once and generate every sklearn estimator:

.. code-block:: python

   from nampy.models import estimator_family
   from nampy.neural.registry import NeuralArchitecture, register_architecture

   register_architecture(NeuralArchitecture(
       name="my_model",
       estimator_prefix="MyModel",
       module_path=f"{__name__}:MyCustomModel",
       config_path=f"{__name__}:MyModelConfig",
       capabilities=frozenset({
           "regression",
           "classification",
           "distributional",
       }),
   ))
   estimators = estimator_family("my_model", module_name=__name__)
   MyRegressor = estimators.regressor
   MyClassifier = estimators.classifier
   MyLSS = estimators.lss

4. Use Your Custom Model
~~~~~~~~~~~~~~~~~~~~~~~~~

Now use it like any other NAMpy model:

.. code-block:: python

   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split
   
   # Generate data
   X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
   
   # Train
   model = MyRegressor(
       numerical_preprocessing="standardization",
       hidden_size=256,
       num_layers=4
   )
   
   model.fit(X_train, y_train, max_epochs=20, lr=1e-3, random_state=7)
   
   # Predict
   predictions = model.predict(X_test)
   
   # Evaluate
   score = model.score(X_test, y_test)
   print(f"R² Score: {score:.4f}")

Complete Example
----------------

This example projects each numerical or categorical feature block into a token
and attends across those tokens. It supports numerical-only, categorical-only,
and mixed inputs. One-hot blocks keep their full encoded width. Its output is
a global contextual contribution, not a collection of marginal feature effects:
attention can make every token depend on the other tokens.

.. code-block:: python

   from dataclasses import dataclass
   from nampy.neural.architectures import BaseModel
   from nampy.models import estimator_family
   from nampy.neural.registry import NeuralArchitecture, register_architecture
   import torch
   import torch.nn as nn
   
   # 1. Configuration
   @dataclass
   class AttentiveMLPConfig:
       lr: float = 1e-4
       lr_patience: int = 10
       weight_decay: float = 1e-6
       lr_factor: float = 0.1
       hidden_size: int = 128
       num_heads: int = 4
       dropout: float = 0.3
   
   # 2. Model Implementation
   class AttentiveMLP(BaseModel):
       def __init__(
           self,
           cat_feature_info,
           num_feature_info,
           num_classes: int = 1,
           config=None,
           **kwargs,
       ):
           super().__init__(**kwargs)
           self.save_hyperparameters(ignore=["cat_feature_info", "num_feature_info"])
           
           if config is None:
               config = AttentiveMLPConfig()
           # Each source feature is a token, even when its encoded block
           # spans several columns. Either feature dictionary may be empty.
           self.feature_names = list(num_feature_info) + list(cat_feature_info)
           infos = list(num_feature_info.values()) + list(cat_feature_info.values())
           if not infos:
               raise ValueError("At least one input feature is required.")
           self.projections = nn.ModuleList([
               nn.Linear(int(info.get("dimension", 1)), config.hidden_size)
               for info in infos
           ])
           self.global_term_names = ("global_attention",)

           # Multi-head attention
           self.attention = nn.MultiheadAttention(
               embed_dim=config.hidden_size,
               num_heads=config.num_heads,
               dropout=config.dropout,
               batch_first=True
           )
           
           # Feed-forward network
           self.ffn = nn.Sequential(
               nn.Linear(config.hidden_size, config.hidden_size * 4),
               nn.ReLU(),
               nn.Dropout(config.dropout),
               nn.Linear(config.hidden_size * 4, config.hidden_size),
               nn.Dropout(config.dropout)
           )
           
           # Output layer
           self.output_layer = nn.Linear(config.hidden_size, num_classes)
       
       def forward(self, num_features: dict, cat_features: dict) -> dict:
           inputs = {**num_features, **cat_features}
           tokens = [
               projection(inputs[name].float())
               for name, projection in zip(self.feature_names, self.projections)
           ]
           x = torch.stack(tokens, dim=1)  # batch, source features, hidden size

           # Self-attention
           attn_out, _ = self.attention(x, x, x)
           x = x + attn_out  # Residual connection
           
           # Feed-forward network
           ffn_out = self.ffn(x)
           x = x + ffn_out  # Residual connection
           
           # Output
           x = x.mean(dim=1)  # Pool the contextual feature tokens
           output = self.output_layer(x)
           
           return {"output": output, "global_attention": output}
   
   # 3. One declaration generates every supported objective surface
   register_architecture(NeuralArchitecture(
       name="attentive_mlp",
       estimator_prefix="AttentiveMLP",
       module_path=f"{__name__}:AttentiveMLP",
       config_path=f"{__name__}:AttentiveMLPConfig",
       capabilities=frozenset({
           "regression", "classification", "distributional"
       }),
   ))
   AttentiveMLPRegressor = estimator_family(
       "attentive_mlp", module_name=__name__
   ).regressor
   
   # 4. Usage
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split

   rng = np.random.default_rng(7)
   X = pd.DataFrame({"x": np.linspace(-1, 1, 160), "z": rng.normal(size=160)})
   y = np.sin(3 * X["x"]) + 0.2 * X["z"] + rng.normal(0, 0.1, len(X))
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)

   model = AttentiveMLPRegressor(
       numerical_preprocessing="standardization",
       hidden_size=64,
       num_heads=4
   )
   
   model.fit(X_train, y_train, max_epochs=20, lr=1e-3, random_state=7)
   predictions = model.predict(X_test)

Important Notes
---------------

Forward Pass Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

The `forward()` method MUST:

* Accept `num_features` and `cat_features` as dictionaries
* Return a dictionary with at least an ``"output"`` key
* The output shape should be (batch_size, num_classes)

For Interpretable Models
~~~~~~~~~~~~~~~~~~~~~~~~

If you want feature-level predictions (like NAM):

.. code-block:: python

   def forward(self, num_features: dict, cat_features: dict) -> dict:
       # Process each feature separately
       feature_outputs = {}
       
       for feature_name, feature_tensor in num_features.items():
           feature_outputs[feature_name] = self.feature_nets[feature_name](feature_tensor)
       
       # Sum for final prediction
       output = sum(feature_outputs.values())
       
       return {
           "output": output,
           **feature_outputs  # Flat named tensors form the additive contract
       }

Configuration Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always include these common hyperparameters:

* ``lr`` - Learning rate
* ``lr_patience`` - Learning rate scheduler patience
* ``weight_decay`` - L2 regularization
* ``lr_factor`` - Learning rate reduction factor

Testing Your Model
------------------

Test on small synthetic data first:

.. code-block:: python

   from sklearn.datasets import make_regression
   
   # Small test
   X, y = make_regression(n_samples=100, n_features=5, random_state=42)
   
   model = MyRegressor()
   model.fit(X, y, max_epochs=10, lr=1e-3)
   
   # Should complete without errors
   predictions = model.predict(X)
   print(f"Output shape: {predictions.shape}")
   print(f"Score: {model.score(X, y):.4f}")

Contributing Your Model
-----------------------

If your model is stable and documented, consider contributing it to NAMpy.

See :doc:`../contributing` for guidelines.

Resources
---------

* :class:`nampy.neural.architectures.BaseModel` - Base model class
* :class:`nampy.neural.registry.NeuralArchitecture` - architecture declaration
* :func:`nampy.models.estimator_family` - generated estimator surfaces
* Existing models in `nampy/neural/architectures/` for reference
