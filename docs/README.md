# NAMpy Documentation

This directory contains the Sphinx documentation for NAMpy.

## Building the Documentation

### Prerequisites

Install Pandoc with your system package manager (for example,
`sudo apt-get install pandoc` on Ubuntu), then install the Python documentation
dependencies:

```bash
pip install -e ".[docs]"
```

Or install from documentation requirements:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

```bash
# From the docs/ directory
make html

# Or from the project root
make docs
```

The built documentation will be in `docs/_build/html/`. Open `index.html` in your browser.

### Build and Serve Locally

```bash
# From project root
make docs-serve

# Then open http://localhost:8000 in your browser
```

### Clean Build

```bash
make clean
```

## Documentation Structure

```text
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation page
├── compatibility.rst    # Public mgcv/scam compatibility scope
├── tutorials/           # Question-led analyses with generated figures
├── howto/              # Focused modeling recipes
├── concepts/           # Statistical and interpretation background
├── notebooks/           # Executed teaching notebooks
├── user_guide/         # Practical guides linked from howto/
├── api/                 # Public API contracts and generated references
├── models/index.rst     # Compact model-selection catalog
├── examples/            # Short copyable task recipes
├── contributing.rst     # Contributing guide
├── development/         # Numerical boundaries and executed validation reports
├── license.rst          # License information
├── faq.rst             # FAQ
├── _static/            # Static files (CSS, images)
├── scripts/            # Reproducible documentation-figure generators
├── _templates/         # Custom templates
└── _build/             # Built documentation (gitignored)
```

The supported mgcv/SCAM selection routes and their intentional upstream
boundaries are recorded in
[development/mgcv_optimizer_boundaries.md](development/mgcv_optimizer_boundaries.md).

The teaching notebooks in `notebooks/` are rendered with saved outputs. Broad
architecture/task sweeps live in `development/notebooks/` as validation reports.
Execute both from clean kernels with:

```bash
python docs/scripts/execute_notebooks.py
python docs/scripts/execute_notebooks.py docs/development/notebooks/*.ipynb
python docs/scripts/generate_optimizer_matrix.py
```

Regenerate the homepage and tutorial-gallery figures from public
NAMpy APIs with:

```bash
MPLCONFIGDIR=/tmp/nampy-matplotlib \
  python docs/scripts/generate_first_gam_assets.py
```

Autosummary may create files below `api/generated/` during a build. That
directory is ignored and must not be committed. `docs/_ext/public_reference.py`
generates effective constructor defaults, training controls, and family
contracts from the installed API into `_generated/reference/`. The shared
navigation excludes generated symbols; users access them from reference pages.

After building, run `python docs/scripts/check_documentation.py` to check local
links, anchors, redirects, navigation consistency, and notebook records.

## Writing Documentation

### reStructuredText (RST) Basics

NAMpy documentation uses reStructuredText. Quick reference:

#### Headings

```rst
Main Heading
============

Section
-------

Subsection
~~~~~~~~~~
```

#### Code Blocks

```rst
.. code-block:: python

   from nampy.models import NAMRegressor
   model = NAMRegressor()
```

#### Links

```rst
:doc:`other_page`  # Link to another doc
:class:`nampy.models.NAMRegressor`  # Link to class
:meth:`fit`  # Link to method
```

#### Lists

```rst
* Item 1
* Item 2

1. First
2. Second
```

### API Documentation

API documentation is auto-generated from docstrings using Sphinx autodoc.
Docstrings should follow NumPy style:

```python
def function(param1, param2):
    """
    Brief description.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    type
        Description of return value.

    Examples
    --------
    >>> function(1, 2)
    3
    """
```

## Contributing

When adding new features:

1. Update relevant RST files
2. Add docstrings to new classes/functions
3. Add model theory to the model reference and a question-led analysis to
   ``tutorials/``; keep focused procedures in ``howto/``
4. Build and check locally before committing

## Resources

* [Sphinx Documentation](https://www.sphinx-doc.org/)
* [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
* [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

The introductory tutorial has a core regression walkthrough followed by optional
classification, calibration, and cross-validation extensions. Real-data
classification is the practical continuation before branching into statistical,
neural, or distributional detailed examples.

Homepage and Modeling Workflows code comes from
`_includes/model_showcases.inc`; regenerate its output with
`_scripts/generate_landing_gam_results.py` and
`_scripts/generate_landing_nam_results.py`. The detailed tree/quantile and practical
workflow generators under `scripts/` emit displayed tables and plots together.
The execution helper records full commits plus source-content manifests and
removes only incidental display noise, preserving results and meaningful warnings.
