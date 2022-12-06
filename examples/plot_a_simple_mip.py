"""
=============================
A simple MIP
=============================
.. NOTE:: Example taken from `gurobi examples <https://assets.gurobi.com/pdfs/user-events/2017-frankfurt/Modeling-1.pdf>`_.

This example demostrates how to use the low-sugar in combination with mlflow
for a very simple single objective MIP experiment tracking

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_low_sugar.ipynb

"""
# sphinx_gallery_thumbnail_path = '_static/a_simple_mip.png'
import datetime
import gurobipy as gp
import mlflow
from mlflow import MlflowException

# import sys; sys.path.append('/Users/Juan.ChaconLeon/opt/opt-sugar/src')  # when running locally
from opt_sugar import low_sugar
from opt_sugar import opt_flow

try:
    experiment_name = f"opt_exp_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):

    def build(data):
        # Not using data in this simple example:
        del data

        # Create a new model
        m = gp.Model("mip1")

        # Create variables
        x = m.addVar(vtype='B', name="x")
        y = m.addVar(vtype='B', name="y")
        z = m.addVar(vtype='B', name="z")

        # Set objective
        m.setObjective(x + y + 2 * z, gp.GRB.MAXIMIZE)

        # Add constraint: x + 2 y + 3 z <= 4
        m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

        # Add constraint: x + y >= 1
        m.addConstr(x + y >= 1, "c1")

        # You can Even call mlflow inside this function if within mlfow start_run context manager

        return m

    opt_model = low_sugar.Model(build)
    solution = opt_model.predict(data={})
    model_info = mlflow.sklearn.log_model(opt_model, "opt_model")

# %%
# Load the Registered Model and Optimize with new Data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.

logged_model_uri = model_info.model_uri
print(f"logged_model_uri: {logged_model_uri}")

# Load model as a PyFuncModel.
loaded_model = opt_flow.pyfunc.load_model(logged_model_uri)
solution = loaded_model.optimize(data={})
print(f"solution from the registered model {solution}")


# %%
# .. image:: https://mybinder.org/badge_logo.svg
#  :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_low_sugar.ipynb
