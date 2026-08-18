Custom Models
=============

NAMpy provides a structured way to implement custom models while leveraging
the existing infrastructure.

Overview
--------

To create a custom model, you need to:

1. Define a configuration class
2. Implement the model architecture
3. Create wrapper classes for regression/classification/LSS
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

   from nampy.neural.modules import BaseModel
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
           
           # Calculate input size
           total_input_size = (
               sum([input_shape for input_shape in num_feature_info.values()]) 
               + len(cat_feature_info)
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
           # Concatenate all numerical features
           num_features_tensor = torch.cat(
               [num_features[key] for key in num_features.keys()], 
               dim=1
           )
           
           # Concatenate all categorical features
           cat_features_tensor = torch.cat(
               [cat_features[key] for key in cat_features.keys()], 
               dim=1
           ) if cat_features else torch.empty(num_features_tensor.shape[0], 0)
           
           # Concatenate all features
           input_tensor = torch.cat([num_features_tensor, cat_features_tensor], dim=1)
           
           # Forward pass
           output = self.mlp(input_tensor)
           
           # MUST return a dictionary with "output" key
           return {"output": output}

3. Create Wrapper Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create sklearn-compatible wrappers:

.. code-block:: python

   from nampy.models import (
       SklearnBaseRegressor,
       SklearnBaseClassifier,
       SklearnBaseLSS
   )
   
   class MyRegressor(SklearnBaseRegressor):
       def __init__(self, **kwargs):
           super().__init__(model=MyCustomModel, config=MyModelConfig, **kwargs)
   
   class MyClassifier(SklearnBaseClassifier):
       def __init__(self, **kwargs):
           super().__init__(model=MyCustomModel, config=MyModelConfig, **kwargs)
   
   class MyLSS(SklearnBaseLSS):
       def __init__(self, **kwargs):
           super().__init__(model=MyCustomModel, config=MyModelConfig, **kwargs)

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
   
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3)
   
   # Predict
   predictions = model.predict(X_test)
   
   # Evaluate
   score = model.score(X_test, y_test)
   print(f"R² Score: {score:.4f}")

Complete Example
----------------

Here's a complete working example:

.. code-block:: python

   from dataclasses import dataclass
   from nampy.neural.modules import BaseModel
   from nampy.models import SklearnBaseRegressor
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
           
           total_input_size = (
               sum([input_shape for input_shape in num_feature_info.values()]) 
               + len(cat_feature_info)
           )
           
           # Input projection
           self.input_proj = nn.Linear(total_input_size, config.hidden_size)
           
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
           # Concatenate features
           num_tensor = torch.cat(list(num_features.values()), dim=1)
           cat_tensor = torch.cat(list(cat_features.values()), dim=1) if cat_features else torch.empty(num_tensor.shape[0], 0)
           x = torch.cat([num_tensor, cat_tensor], dim=1)
           
           # Project to hidden size
           x = self.input_proj(x)
           x = x.unsqueeze(1)  # Add sequence dimension
           
           # Self-attention
           attn_out, _ = self.attention(x, x, x)
           x = x + attn_out  # Residual connection
           
           # Feed-forward network
           ffn_out = self.ffn(attn_out)
           x = attn_out + ffn_out  # Residual connection
           
           # Output
           x = x.squeeze(1)  # Remove sequence dimension
           output = self.output_layer(x)
           
           return {"output": output}
   
   # 3. Wrapper Class
   class AttentiveMLPRegressor(SklearnBaseRegressor):
       def __init__(self, **kwargs):
           super().__init__(model=AttentiveMLP, config=AttentiveMLPConfig, **kwargs)
   
   # 4. Usage
   model = AttentiveMLPRegressor(
       numerical_preprocessing="standardization",
       hidden_size=256,
       num_heads=8
   )
   
   model.fit(X_train, y_train, max_epochs=100, lr=1e-3)
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
           "feature_outputs": feature_outputs  # For interpretability
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

* :class:`nampy.neural.modules.BaseModel` - Base model class
* :class:`nampy.models.SklearnBaseRegressor` - Regression wrapper
* :class:`nampy.models.SklearnBaseClassifier` - Classification wrapper
* :class:`nampy.models.SklearnBaseLSS` - LSS wrapper
* Existing models in `nampy/basemodels/` for reference
