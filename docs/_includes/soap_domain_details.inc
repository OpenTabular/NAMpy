Soap films on bounded domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Soap films respect irregular boundaries and holes. The ``so`` basis combines
a smooth boundary interpolant (``sf``) with an interior component (``sw``)
that is zero on the boundary. Each free boundary loop has its own penalty;
the interior has another penalty. ``k`` controls the boundary basis dimension,
while the number of paired interior knots controls the interior dimension.

Pass boundary loops through ``xt['bnd']``, with coordinate names matching the
formula. Each loop must describe a closed polygon; an omitted closing point is
added automatically. Supply interior knot coordinates through the model's
``knots`` argument. For example:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from nampy.gam import GAM

   rng = np.random.default_rng(42)
   data = pd.DataFrame(rng.uniform(0.05, 0.95, (150, 2)), columns=['x', 'z'])
   data['y'] = np.sin(4 * data.x) * np.cos(3 * data.z)
   boundary = [{'x': [0., 1., 1., 0., 0.],
                'z': [0., 0., 1., 1., 0.]}]
   gx, gz = np.meshgrid([0.2, 0.5, 0.8], [0.2, 0.5, 0.8])
   knots = {'x': gx.ravel(), 'z': gz.ravel()}
   xt = {'bnd': boundary, 'nmax': 40}
   model = GAM(
       formula=f"y ~ s(x, z, bs='so', k=8, xt={xt!r})",
       knots=knots, smoothing_method='reml', optimize_smoothing=True,
   ).fit(data=data)
   prediction, standard_error = model.predict(data, return_se=True)

Use additional loops to describe holes, and a list of ``k`` values to give
each loop a different boundary dimension. ``xt['nmax']`` controls the
finite-difference grid resolution (default 200); changing it changes the
numerical basis. ``xt['bndSpec']`` selects the cyclic boundary spline, using
``{'bs': 'cc'}`` by default or ``{'bs': 'cp', 'm': [2, 2]}``.

An optional ``f`` array on a boundary loop supplies known boundary values.
These become a fixed smooth-owned offset for ``so``/``sf``, including during
new-data prediction and component decomposition. They are added to any formula
or caller-supplied offset. ``sw`` has a zero boundary and does not use this
offset. A boundary-only ``sf`` term needs at least one free boundary loop.

Training observations outside the domain are rejected; prediction rows outside
the domain return NaNs. ``so`` drops outside knots with a warning, following
mgcv; ``sf`` and ``sw`` retain upstream knot validation. Soap compositions
depend on the constructed penalty count:

* ``te``, ``ti`` and ``t2`` accept ``sw``, ``sf`` with exactly one free boundary
  loop, and ``so`` when every boundary loop has prescribed values. Fully known
  ``so`` has only the full-rank interior penalty; holes are supported.
* ``fs`` accepts single-penalty ``sf``. ``sz`` accepts single-penalty ``sf``,
  ``sw`` and fully known-boundary ``so``.
* ``so`` with any free boundary and ``sf`` with multiple free loops have
  multiple penalties. Upstream tensor and factor constructors reject these
  combinations. ``sf`` with no free loops has no free coefficients.
* Full-rank ``sw`` or fully known ``so`` inside a real ``fs`` interaction
  reaches an upstream zero-null-space indexing failure; NAMpy rejects it
  explicitly. This restriction does not apply to ``sz``.

For example, ``te(x,z,t,bs=['so','cr'],d=[2,1],k=[8,5],
xt=[{'bnd': known_boundary},None])`` uses a fully prescribed soap boundary;
supply the soap knots as for a direct term. A factor example is
``s(x,z,group,bs='fs',xt={'bs':'sf','bnd':boundary})``.

The upstream tensor and factor constructors discard the soap-owned boundary
offset. Nested terms follow that behavior even when the standalone soap term
would have a nonzero fixed boundary contribution. In contrast, ``fs`` or
``sz`` with no factor delegates to the ordinary soap constructor and retains
its offsets and explicit term smoothing parameters.

Matrix-valued paired coordinates and summation weights are supported for
``fs(sf)``, ``sz(sf)``, ``sz(sw)`` and fully known ``sz(so)`` with scalar
categorical factors. For example, use ``s(X,Y,group,bs='fs',by=L,
xt={'bs':'sf','bnd':boundary})`` with array-valued ``X``, ``Y`` and ``L``
cells and a categorical ``group`` column. Boundary and knot names must match
the coordinate names. A group that varies across quadrature columns can instead
use explicit categorical rows, as described below. Classless character arrays
are not silently converted to factors. Analytic soap derivatives remain unavailable.
See ``examples/example_soap_compositions.py`` for runnable recipes.
The static reference tests cover basis and penalty matrices, domains with
holes, known boundaries, fitting, prediction uncertainty, and persistence.

The newer ``sc``/``scad`` codes are distinct from SCAM's ``mpi`` and related
bases below. Their default, unconstrained branches reproduce ``bs`` and
adaptive P-splines respectively. Requests such as ``xt=['m+']`` raise
``NotImplementedError``: NAMpy does not run mgcv's separate inequality-based
``scasm`` fitter. Use the documented SCAM bases for supported constrained fits.
