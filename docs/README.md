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
├── getting_started.rst  # Installation and first-analysis routes
├── installation.rst     # Installation guide
├── quickstart.rst       # First statistical GAM analysis
├── neural_quickstart.rst # First neural additive analysis
├── user_guide.rst       # Modeling and interpretation topics
├── examples/index.rst   # Tutorials, recipes, and extended notebooks
├── api/index.rst        # Signature-level API reference
├── about.rst            # Project and development information
├── compatibility.rst    # Public mgcv/scam compatibility scope
├── tutorials/           # Question-led analyses with generated figures
├── notebooks/            # Broad reference notebooks
├── howto/               # Focused recipes listed in the examples gallery
├── user_guide/          # Detailed modeling reference
├── api/                 # Generated and hand-written API pages
├── models/index.rst     # Beginner model choice and architecture comparison
├── contributing.rst     # Contributing guide
├── development/         # Reference-source and optimizer-boundary notes
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

The four notebooks in `notebooks/` are rendered by nbsphinx/nbconvert with
Pandoc. They are broad surveys rather than the primary learning path. Longer
terminal-runnable programs live in `../examples/`; Sphinx example pages
contain shorter task-based recipes.

Regenerate the homepage, quickstart, and tutorial-gallery figures from public
NAMpy APIs with:

```bash
MPLCONFIGDIR=/tmp/nampy-matplotlib \
  python docs/scripts/generate_first_gam_assets.py
```

Autosummary may create files below `api/generated/` during a build. That
directory is ignored and must not be committed; its pages are derived from
public docstrings rather than maintained as another documentation source.

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
