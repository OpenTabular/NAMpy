Select interactions from validation data
========================================

An additive model can miss a relationship that depends jointly on two inputs. We will first fit a main-effects model, inspect its validation residuals, and then use SIAN to select interaction terms.

The test outcomes are reserved for the final comparison; they are not used to choose interactions.

Generate data with an interaction
---------------------------------

.. code-block:: python

   import tempfile
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error, r2_score
   from nampy.models import NAMRegressor, SIANRegressor

   rng = np.random.default_rng(43)
   X = pd.DataFrame(rng.uniform(-1, 1, (900, 3)), columns=["x", "z", "w"])
   y = np.sin(3*X.x) + .4*X.z + 2.5*X.x*X.z + rng.normal(0, .12, len(X))
   X_fit, X_test, y_fit, y_test = train_test_split(X, y, test_size=.2, random_state=43)
   X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit, test_size=.25, random_state=44)
   fit_options = dict(max_epochs=100, lr=.01, batch_size=128, patience=15,
                      random_state=43, checkpoint_path=tempfile.mkdtemp(),
                      logger=False, enable_progress_bar=False, enable_model_summary=False)

Fit a main-effects model
------------------------

.. code-block:: python

   baseline = NAMRegressor(layer_sizes=[24, 12], dropout=0., numerical_preprocessing="minmax")
   baseline.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_options)

Inspect validation residuals
----------------------------

.. code-block:: python

   residual = y_val.to_numpy()-baseline.predict(X_val)
   residual_association = np.corrcoef(residual, X_val.x*X_val.z)[0,1]
   residual_figure, residual_axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
   residual_axis.scatter(X_val.x*X_val.z, residual, s=10, alpha=.4)
   residual_axis.set(xlabel="Validation x*z", ylabel="Baseline residual",
                     title=f"Residual/product correlation: {residual_association:.3f}")

.. image:: /_static/interaction_selection_residuals.png
   :alt: Validation residuals against the candidate x times z interaction, labeled with their correlation
   :class: scientific-figure

An association between residuals and the product x*z is evidence to investigate
a joint term. The simulation supplies a known interaction; this visual/score
check alone does not prove that an interaction is needed in an observational
analysis.

Select interaction terms
------------------------

SIAN fits a reference network on training rows and scores candidate subsets
using validation inputs. Explicitly setting ``interactions`` or
``interaction_degree`` would bypass discovery. Here both remain unset; the
maximum candidate order is two. The residual network is disabled so the
interpretation concerns the reported additive terms.

.. code-block:: python

   selected = SIANRegressor(
       layer_sizes=[24, 12], residual_network=False,
       numerical_preprocessing="minmax", max_interaction_order=2,
       interaction_thresholds=.1, reference_layer_sizes=[32, 16],
       reference_epochs=80, selection_max_samples=80,
   )
   selected.fit(X_train, y_train, X_val=X_val, y_val=y_val, **fit_options)
   selection_table = selected.interaction_selection_table()

.. include:: /_generated/interaction_detector.inc

The selection table records retained candidate interactions. The selection
score summarizes the discovery procedure's evidence; it is not a p-value or
causal importance.


Compare test predictions
------------------------

.. code-block:: python

   interaction_results = pd.DataFrame([
       {"Model": name, "Test n": len(X_test),
        "Test RMSE (lower better)": np.sqrt(mean_squared_error(y_test, model.predict(X_test))),
        "Test R2 (higher better)": r2_score(y_test, model.predict(X_test))}
       for name, model in [("Main effects", baseline), ("Selected SIAN", selected)]
   ])

.. include:: /_generated/interaction_selection_results.inc

Inspect the selected interaction
--------------------------------

.. code-block:: python

   grid_values = np.linspace(-1, 1, 45)
   gx, gz = np.meshgrid(grid_values, grid_values)
   grid = pd.DataFrame({"x": gx.ravel(), "z": gz.ravel(), "w": 0.})
   parts = selected.predict_components(grid, center=True, reference_X=X_train)
   parts.validate_additive_reconstruction(atol=1e-5)
   interaction_terms = [name for name, meta in parts.term_metadata.items() if meta.kind == "interaction" and set(meta.features) == {"x", "z"}]
   interaction_surface = sum((np.asarray(parts.terms[name]).reshape(-1) for name in interaction_terms), np.zeros(len(grid)))
   figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
   image = axis.pcolormesh(gx, gz, interaction_surface.reshape(gx.shape), shading="auto")
   axis.set(xlabel="x", ylabel="z", title="Selected x:z contribution" if interaction_terms else "No x:z term selected")
   figure.colorbar(image, ax=axis, label="Centered link contribution")

.. image:: /_static/interaction_selection_workflow.png
   :alt: Selected centered x:z interaction contribution with w held at zero
   :class: scientific-figure

The surface contains only selected x:z contributions, with w held at zero;
it excludes the intercept and main effects. A blank surface means no such
term was selected. Residual evidence and discovery used validation rows;
the performance table uses final test rows. This split does not establish
selection stability or a unique separation between overlapping effects.

The :doc:`interpretability guide <../user_guide/interpretability>` explains
centering and overlapping term contributions. The :doc:`SIAN model guide
<../models/sian>` describes the selection controls and optional residual network.
