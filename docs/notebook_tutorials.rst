Notebook Tutorials
==================

``docs/notebooks/`` is the single canonical notebook collection. The Sphinx
build regenerates these deterministic ``.ipynb`` sources from
``docs/generate_notebooks.py`` before nbsphinx and Pandoc render them. This
keeps the checked-in notebooks reviewable while preventing generated content
from becoming stale.

Start with the overview, choose the statistical GAM case study, or open the
notebook for a neural architecture:

.. toctree::
   :maxdepth: 1
   :caption: Theory-first tutorials

   notebooks/00_overview
   notebooks/01_gam
   notebooks/02_linreg
   notebooks/03_nam
   notebooks/04_snam
   notebooks/05_sian
   notebooks/06_gpnam
   notebooks/07_igann
   notebooks/08_nbm
   notebooks/09_spam
   notebooks/10_nbm_spam
   notebooks/11_treenam
   notebooks/12_ensemble_treenam
   notebooks/13_nodegam
   notebooks/14_natt
   notebooks/15_namformer
   notebooks/16_qnam
   notebooks/17_spline_nam
   notebooks/18_neural_ensemble

The notebooks default to a no-training path so imports, constructors, and
public calls can be inspected quickly. Enable the documented training flags to
run fitted examples. Their checked-in outputs are intentionally empty; tests
verify generator parity, compile every code cell, exercise the default path,
and run the primary fitted GAM story.

Run locally
-----------

.. code-block:: bash

   pip install -e ".[all,docs]" jupyterlab
   jupyter lab docs/notebooks

Short copy-and-paste recipes remain in :doc:`examples/index`; terminal-runnable
verification programs remain in the root ``examples/`` directory.
