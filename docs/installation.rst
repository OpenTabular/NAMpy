Installation
============

.. important::

   **Requirements:** Python 3.11 or 3.12. PyTorch is installed with NAMpy.

Quick install
-------------

.. code-block:: bash

   pip install nampy

This installs the statistical GAM and neural modeling backends together with
their runtime dependencies.

Verify the installation
-----------------------

.. code-block:: python

   import nampy

   print(nampy.__version__)

GPU support
-----------

Statistical GAMs run on the CPU. Neural estimators use PyTorch and can train on
a supported GPU. Check whether CUDA is available with:

.. code-block:: python

   import torch

   print(torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))

The PyTorch build installed from PyPI may not match the CUDA runtime on every
machine. If CUDA is unavailable, select the appropriate command with the
`PyTorch installation guide <https://pytorch.org/get-started/locally/>`_, then
install NAMpy.

.. note::

   A GPU is optional. It is most useful for larger neural models and datasets;
   the package also runs on the CPU.

Troubleshooting
---------------

First confirm that the Python interpreter running your script can see the same
installation:

.. code-block:: bash

   python -m pip show nampy

If installation or import still fails, search the
`issue tracker <https://github.com/OpenTabular/NAMpy/issues>`_ or open a report
with the Python version, operating system, installation command, and complete
error message.

Next steps
----------

* :doc:`quickstart` — fit and interpret a statistical GAM.
* :doc:`neural_quickstart` — fit and interpret a neural additive model.
* :doc:`user_guide` — understand the modeling and interpretation workflow.
* :doc:`faq` — find answers to common setup and modeling questions.
