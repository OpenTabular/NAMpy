Installation
============

Requirements
------------

NAMpy supports Python 3.11 and 3.12. The core package requires:

* scikit-learn
* pandas
* numpy (<=1.26.4)

Backend dependencies are grouped into ``gam`` and ``neural`` installation
extras. The ``all`` extra installs both.

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install both supported backends with:

.. code-block:: bash

   pip install "nampy[all]"

Install one backend when a smaller environment is preferable:

.. code-block:: bash

   pip install "nampy[gam]"
   pip install "nampy[neural]"

From Source
~~~~~~~~~~~

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/Ananyapam7/NAMpy.git
   cd NAMpy
   pip install -e ".[all]"

From GitHub
~~~~~~~~~~~

You can also install directly from Github:

.. code-block:: bash

   pip install "nampy[all] @ git+https://github.com/Ananyapam7/NAMpy.git@main"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

If you want to contribute to NAMpy, install with development dependencies:

.. code-block:: bash

   git clone https://github.com/Ananyapam7/NAMpy.git
   cd NAMpy
   pip install -e ".[all,dev]"
   
   # Optional: install the repository's local quality hooks
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

* Check the `GitHub Issues <https://github.com/Ananyapam7/NAMpy/issues>`_
* Ask a question in `GitHub Discussions <https://github.com/Ananyapam7/NAMpy/discussions>`_
* Consult the :doc:`faq` page
