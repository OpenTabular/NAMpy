Architecture
============

NAMpy separates fitted estimators from their numerical backends:

.. code-block:: text

   nampy.models                  public sklearn-style estimator facade
       |-- neural estimators --> nampy.neural
       |                         |-- architectures/  torch forward models
       |                         |-- objectives.py   output/loss semantics
       |                         |-- distributions/  LSS families
       |                         `-- data/, task.py  training runtime
       `-- GAM adapters -------> nampy.gam           mgcv-parity backend

``nampy.models`` is therefore the generic public layer. ``nampy.neural`` and
``nampy.gam`` remain separate because they use fundamentally different fitting
algorithms. Both expose additive predictions through the shared contracts and
plotting surfaces.

Neural architecture contract
----------------------------

A neural architecture always defines a forward model. It accepts transformed
feature dictionaries and an output width, then returns a dictionary containing
an ``"output"`` tensor. Ordinarily it does not choose a loss, encode targets,
or interpret the output. Architectures with a published non-gradient optimizer
may additionally implement the native-training contract described below.

Each architecture is declared once with
:class:`nampy.neural.registry.NeuralArchitecture`. Its capability set controls
which public estimators are generated. The declaration can also own
architecture-specific preprocessing defaults and input requirements. Regression,
binary/multiclass, and LSS behavior comes from objective objects rather than
architecture subclasses.

.. code-block:: text

   architecture + objective/distribution = fitted neural estimator

For LSS, the selected distribution determines the required output width,
target representation, per-row loss, prediction transform, and metrics. Any
registered architecture with the ``"distributional"`` capability therefore
supports every registered compatible distribution without an LSS-specific
architecture implementation.

Optional training and inference contracts
-----------------------------------------

Staged estimators can resolve a train-data-dependent architecture configuration
after the shared train/validation split and train-only preprocessing, but before
the final Torch module is constructed. Registry-provided estimator mixins own
this lifecycle step; the forward architecture remains objective- and
data-loader-independent. SIAN uses it for reference-model interaction
discovery, and stores the resolved copy as ``fitted_config_`` without mutating
the constructor configuration used by cloning and parameter search.

Interaction detection itself lives under
``nampy.neural.interaction_selection``. Detectors consume grouped transformed
features and implement a common score contract; hierarchical frontier
construction and thresholding are separate from the detector. The generic
block-masked additive component similarly remains independent of SIAN and can
be reused by equal-depth explicit-term architectures.

Architectures that are linear over a fixed basis can implement the
``FixedLinearDesignProvider`` protocol. The shared estimator engine then allows
a strict conjugate-gradient regression solve without adding solver logic to the
public wrapper. GPNAM is the first implementation of this contract. Its
classification and distributional estimators still use the ordinary objective
engine because those objectives are not the released least-squares solve.

Architectures with a complete non-gradient fitting algorithm can instead
implement ``NativeTrainingProvider`` and declare ``"native_training"``.
The estimator facade still owns splitting, train-only preprocessing, target
encoding, offsets, persistence, batched inference, and public result objects;
the architecture receives prepared train/validation tensors and owns only its
optimizer. IGANN uses this contract for its sequential Newton/ELM ridge solves.
Native diagnostics are exposed uniformly through ``native_training_info_`` and
``training_history()``.

The shared estimator facade also owns seeded model construction, batched
inference, and total/trainable parameter diagnostics. Fixed-basis architectures
may additionally expose ``basis_transform`` and ``basis_metadata``.
Architectures can augment ``model_complexity()`` with effective fitted counts
through ``complexity_metadata``.

Adding an architecture
----------------------

An architecture contribution consists of:

1. one class under ``nampy/neural/architectures/``;
2. one configuration dataclass under ``nampy/neural/configs/``;
3. one :class:`~nampy.neural.registry.NeuralArchitecture` declaration;
4. architecture-focused tests.

The :func:`nampy.models.estimator_family` factory generates every estimator
surface allowed by the declaration. Unsupported surfaces, such as
classification for ``SplineNAM``, are represented explicitly by capabilities
instead of being inferred from missing wrapper files.

GAM isolation
-------------

The GAM backend intentionally does not reuse Torch distribution objectives.
An mgcv family participates in PIRLS, derivative calculation, smoothing
selection, and diagnostics, so it is part of the fitting engine rather than a
replaceable neural output head. The two backends share public result contracts,
not numerical family implementations.
