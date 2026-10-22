Neural additive models
======================

The modeling problem
--------------------

Neural additive models learn nonlinear feature functions with neural networks
while retaining named contributions that can be inspected after fitting. They
are useful when learned representations are desirable but a single unrestricted
prediction function would hide how individual variables contribute.

Start with NAM
--------------

NAM is the clearest first neural model: one feature network learns the effect
of each input. The introductory example uses the same synthetic electricity
demand data as the statistical GAM example, so the estimator—not the meaning
of the variables—is the only new idea.

.. code-block:: python

   from nampy.models import NAMRegressor

   model = NAMRegressor(layer_sizes=[48, 24], random_state=21)
   model.fit(X_train, y_train, max_epochs=80, patience=10)

   predictions = model.predict(X_test)
   effects = model.plot_terms(X_test, reference_X=X_train, center=True)

The reported temperature contribution depends only on temperature. Explicit
interaction terms depend on all variables named by that term. Centering fixes
a reference for plotting; it does not prove that effects are uniquely
recoverable when predictors are correlated.

One shared workflow
-------------------

Neural estimators share preprocessing, ``fit`` and ``predict``, named
component prediction, term plots, persistence, and regression,
classification, or distributional task wrappers where supported. Architecture
pages therefore focus on what changes: the representation of feature
functions, interaction structure, sparsity mechanism, and interpretation
restrictions.

Choose a capability first
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 13 13 14 15

   * - Starting point
     - Regression
     - Classification
     - Distributional
     - Interactions
   * - NAM: separate feature networks
     - Yes
     - Yes
     - Yes
     - Explicit
   * - NBM: shared learned bases
     - Yes
     - Yes
     - Yes
     - Explicit
   * - SNAM: sparse feature selection
     - Yes
     - Yes
     - Yes
     - Explicit
   * - SIAN: sparse interaction discovery
     - Yes
     - Yes
     - Yes
     - Discovered
   * - GPNAM: approximate GP effects
     - Yes
     - Yes
     - Yes
     - Explicit
   * - SPAM / NBM-SPAM: polynomial structure
     - Yes
     - Yes
     - Yes
     - Learned

This table describes structural support, not expected performance. The
:doc:`complete architecture comparison <../models/index>` covers tree,
transformer, spline, quantile, and other models, with references and
interpretation limits.

Continue
--------

* Run :doc:`your first neural additive model
  <../auto_examples/neural/plot_first_nam>`.
* Compare :doc:`NAM and NBM <../tutorials/nam_nbm>` on one plotting scale.
* Use :doc:`../user_guide/training` for optimization and validation controls.
* Look up public estimators in :doc:`../api/models`.
