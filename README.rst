.. -*- mode: rst -*-

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.23.2
.. |GurobiPyMinVersion| replace:: 9.5.2
.. |ScikitLearn| replace:: 1.1.2

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
