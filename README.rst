.. -*- mode: rst -*-

|ReadTheDocs|_ |Pypi|_ |PythonVersion|_ |Black|_

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.23.2
.. |GurobiPyMinVersion| replace:: 9.5.2
.. |ScikitLearn| replace:: 1.1.2

.. |ReadTheDocs| image:: https://readthedocs.org/projects/opt-sugar/badge/?version=latest
.. _ReadTheDocs: https://opt-sugar.readthedocs.io/en/latest/?badge=latest

.. |PyPi| image:: https://img.shields.io/pypi/v/opt-sugar
.. _PyPi: https://pypi.org/project/opt-sugar/

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue
.. _PythonVersion: https://pypi.org/project/scikit-learn/

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/psf/black

**opt-sugar**
is a Python package meant to make the optimization operation (OptOps) tasks easier by providing the building blocks needed
to use mlflow for mathematical optimization experimentation.

The project was started in oct 2022 by Juan Chacon.

Installation
------------

Dependencies
~~~~~~~~~~~~~~~~~

opt-sugar requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- GurobiPy (>= |GurobiPyMinVersion|)
- ScikitLearn (>= |ScikitLearn|)

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation all the dependencies,
the easiest way to install opt-sugar is using ``pip``::

    pip install -U opt-sugar

Read the docs `here. <https://opt-sugar.readthedocs.io/en/latest/>`_

For Contributors
----------------

Releasing Package
~~~~~~~~~~~~~~~~~

To release the package to PyPI follow the next steps:

#. Update version in setup.py file and push changes to GitHub.
#. In GitHub create a tag with the same version as in the setup.py, then create a release using the tag you just created.
#. Have a coffee, the `Upload Python Package` GitHub action will do the rest.

Updating Examples
~~~~~~~~~~~~~~~~~

Once the examples run locally using the imports with local paths in the examples:

#. Remove the sys.path.append lines in the examples (commenting them by now is okay)
#. Release to PyPI
#. Add opt-sugar in the requirements-dev.txt
#. Remove the docs/source/auto_examples folder content
#. Generate the examples locally and push to repo
#. Checkout any changes in the main branch

.. Note::
    It would be good idea to run `pip install -U -r requirements-dev.txt` in your dev environment to get the most
    updated version of the package.
    
