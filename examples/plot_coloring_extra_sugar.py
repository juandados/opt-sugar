"""
=============================
Experiment Tracking: Coloring
=============================

This example demostrates how to use opt-sugar in combination with mlflow
for single objective optimization experiment tracking

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_coloring.ipynb
"""
import datetime
from urllib.parse import urlparse
from itertools import product
from collections import defaultdict
from random import random
import logging

import gurobipy as gp
import mlflow
from mlflow.exceptions import MlflowException


import sys; sys.path.append('/Users/Juan.ChaconLeon/opt/opt-sugar/src')  # when running locally
from opt_sugar.extra_sugar import OptModel, ModelBuilder
from opt_sugar.extra_sugar.objective import Objective, ObjectivePart, BaseObjective
from opt_sugar import opt_flow


# TODO: reformat this example similar to
#  https://github.com/scikit-learn/scikit-learn/blob/main/examples/calibration/plot_calibration_multiclass.py

# %%
# A Colored Graph Helper Function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The following function is really handy to visualize our colored graphs.
from utils.coloring import get_graph_to_show


# %%
# The Optimizations Model Builder
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The following class is the builder for the coloring problem.


class ColoringModelBuilder(ModelBuilder):
    """This should be user implemented"""

    def build_variables(self, base_model):
        degrees = defaultdict(int)
        for v1, v2 in self.data["edges"]:
            degrees[v1] += 1
            degrees[v2] += 1
        self.degree = max(degrees.items(), key=lambda x: x[1])[1]
        color_keys = list(product(self.data["nodes"], range(self.degree)))
        color = base_model.addVars(color_keys, vtype="B", name="color")
        max_color = base_model.addVar(lb=0, ub=self.degree, vtype="C", name="max_color")
        self.variables = {"color": color, "max_color": max_color}

    def build_constraints(self, base_model):
        color = self.variables["color"]
        for v1, c in color:
            # if color[v1, c] == 1 -> color[v2, c] == 0 for all v2 such that (v1, v2) or
            # belongs to E
            for v2 in self.data["nodes"]:
                if (v2, v1) in self.data["edges"] or (v1, v2) in self.data["edges"]:
                    base_model.addConstr(
                        color[v2, c] <= 1 - color[v1, c], name=f"color_{c}_{v1}_{v2}"
                    )

        for v in self.data["nodes"]:
            base_model.addConstr(
                gp.quicksum(color[v, c] for c in range(self.degree)) == 1,
                name=f"every_node_has_color_{v}",
            )

        max_color = self.variables["max_color"]
        for v, c in color:
            base_model.addConstr(
                c * color[v, c] <= max_color, name=f"max_color_{v}_{c}"
            )

    def build_objective(self, base_model):
        max_color = self.variables["max_color"]
        objective_parts = [ObjectivePart(weight=1, expr=max_color)]
        objective = Objective([BaseObjective(objective_parts, hierarchy=1)])
        base_model.setObjective(objective.build()[0], gp.GRB.MINIMIZE)
        return objective


def fit_callback(model):
    fit_callback_data = {
        "mip_gap": model.mip_gap,
        "objective_value": model.getObjective().getValue(),
    }
    return fit_callback_data


# %%
# Generate random data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.

node_count = 16
edge_probability = 0.5

nodes = set(range(node_count))
edges = set(
    (v1, v2)
    for v1, v2 in product(nodes, nodes)
    if random() <= edge_probability and v1 > v2
)
edges.update([(1, 0)])  # Forcing the edge 1, 0 to avoid empty edges
data = {"nodes": nodes, "edges": edges}

# %%
# Tracking an Optimization Experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.

logging.getLogger("mlflow").setLevel(logging.CRITICAL)  # Can be set DEBUG

try:
    experiment_name = f"opt_exp_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    opt_model = OptModel(model_builder=ColoringModelBuilder)
    solution = opt_model.optimize(data, fit_callback)

    # Note: Above is replacement for opt_model.fit(data, fit_callback) and opt_model.predict(data)
    mlflow.log_param("objective_parts", opt_model.objective)
    mlflow.log_metric("kpi", opt_model.fit_callback_data["objective_value"])
    for step, (gap, time) in enumerate(opt_model.log_results.progress('nodelog')[['Gap', 'Time']].values):
        mlflow.log_metric(f"gap", gap, step)
        mlflow.log_metric(f"time", time, step)
    #mlflow.log_metric("nodelog", opt_model.log_results.progress("nodelog"))

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    print(f"tracking_url_type_store: {tracking_url_type_store}")

    # Register the model
    if tracking_url_type_store != "file":
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        model_info = mlflow.sklearn.log_model(
            opt_model, "opt_model", registered_model_name="OptModel"
        )
    else:
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
# Note: the previous line is a replacement opt_flow.pyfunc.load_model

# Data generation
node_count = 6  # Notice this is different input data
edge_probability = 0.5
nodes = set(range(node_count))
edges = set(
    (v1, v2)
    for v1, v2 in product(nodes, nodes)
    if random() <= edge_probability and v1 > v2
)
edges.update([(1, 0)])  # Forcing the edge 1, 0 to avoid empty edges
data = {"nodes": nodes, "edges": edges}

solution = loaded_model.optimize(data)
print(f"Optimized Coloring: {solution}")

g = get_graph_to_show(data, solution)
# g.show(name="vis.html")

# %%
# .. image:: https://mybinder.org/badge_logo.svg
#  :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_coloring.ipynb
