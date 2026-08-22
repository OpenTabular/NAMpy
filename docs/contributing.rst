Contributing
============

NAMpy supports development on Python 3.11 and 3.12. Install the editable package
and development tools with:

.. code-block:: bash

   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[all,dev]"
   pre-commit install

Run the static gates with ``ruff check nampy tests`` and ``mypy nampy``. Tests
should use the smallest relevant pytest slice; the full suite is not the default
development command.

Work under ``nampy/gam/`` must reproduce upstream ``mgcv`` behavior.
Start from a targeted parity test and the corresponding local R/C routine, and
do not introduce approximate or heuristic fallbacks for unsupported behavior.

The complete workflow, test policy, and pull-request checklist are in
`CONTRIBUTING.md <https://github.com/Ananyapam7/NAMpy/blob/main/CONTRIBUTING.md>`_.

Useful links
------------

* `Issue tracker <https://github.com/Ananyapam7/NAMpy/issues>`_
* `Pull requests <https://github.com/Ananyapam7/NAMpy/pulls>`_
* `Discussions <https://github.com/Ananyapam7/NAMpy/discussions>`_
