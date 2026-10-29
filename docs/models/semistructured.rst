SemiStructured
==============

When to use it
--------------

Use this route to combine named linear/spline effects with a jointly learned
neural contribution that is orthogonal to the structured training design.

Model structure
---------------

The predictor is B beta + (H - B A) gamma. B comes from the existing statistical
formula compiler, H is learned, and A is the weighted training projection.
Each step trains the structured and neural parameters jointly. New-data
predictions reuse A. Read :doc:`/user_guide/semistructured` for the mathematical
contract, complete example and unsupported routes.

Fit and inspect
---------------

.. literalinclude:: /../examples/example_semistructured.py
   :language: python

Interpretation and limits
-------------------------

Structured terms retain their source-feature metadata. The neural contribution
is explicitly global over its selected inputs. Orthogonality is empirical on
training rows; it is not a causal interpretation. This native CPU full-cohort
route supports scalar numeric inputs, scalar regression and Normal LSS with
fixed structured penalties. Joint REML, neural coefficient intervals,
classification, categorical/vector inputs and offsets remain unsupported.

Available estimators
--------------------

* :class:`nampy.models.SemiStructuredRegressor`
* :class:`nampy.models.SemiStructuredLSS`

See :doc:`/neural_reference/semistructured` for configuration and task references.
