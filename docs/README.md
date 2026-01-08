# NAMpy Documentation

This directory contains the Sphinx documentation for NAMpy.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

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
├── installation.rst     # Installation guide
├── quickstart.rst       # Quick start guide
├── user_guide.rst       # User guide (overview)
│   └── user_guide/      # Detailed user guides
├── api/                 # API reference
│   ├── index.rst
│   ├── models.rst
│   ├── basemodels.rst
│   ├── preprocessing.rst
│   ├── configs.rst
│   └── utils.rst
├── models/              # Model documentation
│   └── index.rst
├── examples/            # Example notebooks and scripts
│   └── index.rst
├── contributing.rst     # Contributing guide
├── changelog.rst        # Changelog
├── license.rst          # License information
├── faq.rst             # FAQ
├── _static/            # Static files (CSS, images)
├── _templates/         # Custom templates
└── _build/             # Built documentation (gitignored)
```

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

## Read the Docs

The documentation is configured for Read the Docs via `.readthedocs.yaml`.

To build on Read the Docs:

1. Import the project at https://readthedocs.org/
2. The build will use `.readthedocs.yaml` configuration
3. Documentation will be automatically built on each commit

## Continuous Integration

Documentation is automatically built and checked in CI via `.github/workflows/docs.yml`.

## Contributing

When adding new features:

1. Update relevant RST files
2. Add docstrings to new classes/functions
3. Add examples if applicable
4. Build and check locally before committing
5. Update the changelog

## Resources

* [Sphinx Documentation](https://www.sphinx-doc.org/)
* [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
* [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
* [Read the Docs](https://docs.readthedocs.io/)
