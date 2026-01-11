Installation
============

Requirements
------------

NAMpy requires Python 3.6 or higher and the following dependencies:

* PyTorch
* Lightning
* scikit-learn
* pandas
* numpy (<=1.26.4)
* torchmetrics
* properscoring
* matplotlib

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The recommended way to install NAMpy is using pip:

.. code-block:: bash

   pip install nampy

From Source
~~~~~~~~~~~

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/OpenTabular/NAMpy.git
   cd NAMpy
   pip install -e .

From GitHub
~~~~~~~~~~~

You can also install directly from Github:

.. code-block:: bash

   pip install git+https://github.com/OpenTabular/NAMpy.git@main

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

If you want to contribute to NAMpy, install with development dependencies:

.. code-block:: bash

   git clone https://github.com/OpenTabular/NAMpy.git
   cd NAMpy
   pip install -e ".[dev]"
   
   # Optional: Install pre-commit hooks
   pip install pre-commit
   pre-commit install

Verifying Installation
----------------------

To verify that NAMpy is installed correctly:

.. code-block:: python

   import nampy
   print(nampy.__version__)

GPU Support
-----------

NAMpy uses PyTorch as its backend, which supports GPU acceleration. To use GPU:

1. Install PyTorch with CUDA support:

   .. code-block:: bash

      # For CUDA 11.8
      pip install torch --index-url https://download.pytorch.org/whl/cu118

2. Verify GPU availability:

   .. code-block:: python

      import torch
      print(f"CUDA available: {torch.cuda.is_available()}")
      print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

NAMpy models will automatically use GPU if available.

Getting Help
------------

If you encounter issues during installation:

* Check the `GitHub Issues <https://github.com/OpenTabular/NAMpy/issues>`_
* Ask a question in `GitHub Discussions <https://github.com/OpenTabular/NAMpy/discussions>`_
* Consult the :doc:`faq` page
