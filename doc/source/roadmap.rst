.. _roadmap:

.. |ss| raw:: html

   <strike>

.. |se| raw:: html

   </strike>

Roadmap
=======


Purpose of this document
------------------------
This document list general directions that core contributors are interested
to see developed in opt-sugar. The fact that an item is listed here is in
no way a promise that it will happen, as resources are limited. Rather, it
is an indication that help is welcomed on this topic.


Backlog
-------

#. [In Progress] Define a low_sugar submodule, a lighter interface for gurobi models, not requiring that much sweet.

#. Add hierarchical optimization capabilities.

#. Check if the ModelBuilder and OptModel in extra_sugar need to be defined.

#. Include logger results as part of the OptModel object attributes. This can be done using grblogtools.

#. Consider developing mlflow autolog for optimization model `autolog <https://mlflow.org/docs/1.12.1/_modules/mlflow/sklearn.html#autolog>`_.

#. |ss| Add an experiment tracking example gallery. |se|

#. |ss| Modify the ModelBuilder class to have an optimize method that plays well with mlflow. |se|

Examples related
~~~~~~~~~~~~~~~~

#. For Supply Chain (superheros factory) examples:

   * Single objective: Mini toy example (No negative inventories).
   * Single objective: Mini toy example (Negative inventories required).
   * Single objective: Establishing best objective balance using experiment tracking.
   * Multiple Objectives: Priority customers.
   * Add output visualization using pandas.

#. Add examples using the model_builder_factory method.

#. Add examples for the low_sugar submodule.

   * |ss| Add a sudoku example using low_sugar. |se|
   * |ss| Add a coloring example illustrating how to log parameters and metrics. |se|
   * |ss| Add a coloring example illustrating how to run experiments with multiple runs. |se|
   * Add (refactor) the superheros factory examples.

