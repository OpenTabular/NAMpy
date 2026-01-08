Model Reference
===============

This page lists every available model class and its constructor parameters,
including hyperparameters and preprocessing options. Fit-time keyword arguments
are shared across models and documented in the base classes.

Shared Fit-Time Parameters
--------------------------

All models implement a scikit-learn compatible ``fit`` method. For the full list
of training-time keyword arguments (for example ``max_epochs``, ``patience``,
``val_size``, and ``batch_size``), see:

* :class:`nampy.models.SklearnBaseRegressor`
* :class:`nampy.models.SklearnBaseClassifier`
* :class:`nampy.models.SklearnBaseLSS`

.. currentmodule:: nampy.models

Neural Additive Models
----------------------

NAM
~~~

Regressor
^^^^^^^^^

.. autoclass:: NAMRegressor

Classifier
^^^^^^^^^^

.. autoclass:: NAMClassifier

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NAMLSS

GPNAM
~~~~~

Regressor
^^^^^^^^^

.. autoclass:: GPNAMRegressor

Classifier
^^^^^^^^^^

.. autoclass:: GPNAMClassifier

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GPNAMLSS

TreeNAM
~~~~~~~

Regressor
^^^^^^^^^

.. autoclass:: TreeNAMRegressor

SNAM
~~~~

Regressor
^^^^^^^^^

.. autoclass:: SNAMRegressor

NodeGAM
~~~~~~~

Regressor
^^^^^^^^^

.. autoclass:: NodeGAMRegressor

Classifier
^^^^^^^^^^

.. autoclass:: NodeGAMClassifier

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NodeGAMLSS

Basis Function Models
---------------------

NBM
~~~

Regressor
^^^^^^^^^

.. autoclass:: NBMRegressor

Classifier
^^^^^^^^^^

.. autoclass:: NBMClassifier

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NBMLSS

Attention-Based Models
----------------------

NATT
~~~~

Regressor
^^^^^^^^^

.. autoclass:: NATTRegressor

Classifier
^^^^^^^^^^

.. autoclass:: NATTClassifier

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NATTLSS

NAMformer
~~~~~~~~~

Regressor
^^^^^^^^^

.. autoclass:: NAMformerRegressor

Classifier
^^^^^^^^^^

.. autoclass:: NAMformerClassifier

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: NAMformerLSS

Specialized Models
------------------

QNAM (Quantile NAM)
~~~~~~~~~~~~~~~~~~~

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: QNAM

LinReg (Neural Linear Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Regressor
^^^^^^^^^

.. autoclass:: LinRegRegressor

Classifier
^^^^^^^^^^

.. autoclass:: LinRegClassifier

Distributional Regression (LSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LinRegLSS
