import gurobipy as gp
import mlflow
from mlflow import MlflowException
import datetime

# When running locally
from src.opt_sugar.low_sugar import Model
from src.opt_sugar.opt_flow import pyfunc

try:
    experiment_name = f"opt_exp_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):

    def build(data):
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

    opt_model = Model(build)
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
loaded_model = pyfunc.load_model(logged_model_uri)
solution = loaded_model.optimize(data={})
print(f"solution from the registered model {solution}")
