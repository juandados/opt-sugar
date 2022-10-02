.. -*- mode: rst -*-

.. |PythonMinVersion| replace:: 3.8
.. |NumPyMinVersion| replace:: 1.23.2
.. |GurobiPyMinVersion| replace:: 9.5.2
.. |ScikitLearn| replace:: 1.1.2

**opt-sugar**
is a Python package meant to make the optimization operation (OptOps) tasks easier by providing the building block need
to use mlflow for experimentation in the field of mathematical optimization

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

If you already have a working installation of numpy and scipy,
the easiest way to install scikit-learn is using ``pip``::

    pip install -U scikit-learn
