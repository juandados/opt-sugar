"""
===========================================
Experiment Tracking: Coloring Multiple Runs
===========================================

This example demostrates how to use opt-sugar in combination with mlflow
for single objective optimization experiment tracking

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_coloring.ipynb
"""
# sphinx_gallery_thumbnail_path = '_static/coloring_experiment.png'
import datetime
from urllib.parse import urlparse
from itertools import product
import logging
import random

import gurobipy as gp
import mlflow
from mlflow.exceptions import MlflowException

# import sys; sys.path.append('/Users/Juan.ChaconLeon/opt/opt-sugar/src')  # when running locally
from opt_sugar import low_sugar

# The generate_graph_data generates random graphs given a graph size and an edge probability.
from utils.coloring import generate_graph_data

# TODO: reformat this example similar to
#  https://github.com/scikit-learn/scikit-learn/blob/main/examples/calibration/plot_calibration_multiclass.py

# ..NOTE:: check the coloring example first to see how the model was formulated.

from utils.coloring import ColoringModelBuilder

# %%
# Tracking an Optimization Experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.

logging.getLogger("mlflow").setLevel(logging.CRITICAL)  # Can be set DEBUG



def get_build(mip_focus):
    def build(data):
        # Create a new model
        m = gp.Model("coloring")
        model_builder = ColoringModelBuilder(data)
        model_builder.build_variables(m)
        model_builder.build_constraints(m)
        model_builder.build_objective(m)
        # setting parameters
        m.setParam('MIPFocus', mip_focus)
        m.setParam('Method', 4)  # for deterministic concurrent runs
        return m
    return build


def callback(m):
    objective = m.getObjective()
    color_count = objective.getValue()
    callback_result = {"color_count": color_count, "MIPFocus": m.getParamInfo('MIPFocus')[2], "RunTime": m.RunTime}
    return callback_result

# %%
# MIPFocus effect Over Runtime Case 1: Same data multiple runs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.

# Let's first set the random seed for reproducible results
random.seed(42)

try:
    experiment_name = f"coloring_experiment_1_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

# Generating a graph instance
data = generate_graph_data(node_count=15, edge_probability=0.5)

# Building
for mip_focus, _ in product([0, 1, 2, 3], range(25)):
    with mlflow.start_run(experiment_id=experiment_id):
        build = get_build(mip_focus)
        opt_model = low_sugar.Model(build)
        result = opt_model.optimize(data=data, callback=callback)

        # Note: Above is replacement for opt_model.fit(data, fit_callback) and opt_model.predict(data)
        mlflow.log_param("MIPFocus", result["callback_result"]["MIPFocus"])
        mlflow.log_metric("RunTime", result["callback_result"]["RunTime"])
        mlflow.log_metric("color_count", result["callback_result"]["color_count"])

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"tracking_url_type_store: {tracking_url_type_store}")

# %%
# MIPFocus effect Over Runtime Case 1: Different input data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.

# Again Let's first set the random seed for reproducible results
random.seed(21)

try:
    experiment_name = f"coloring_experiment_2_{datetime.datetime.now().strftime('%Y_%m_%d')}"
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

# Building
for mip_focus, data_i in product([0, 1, 2, 3], range(20)):

    data = generate_graph_data(node_count=15, edge_probability=0.5)

    for try_i in range(3):

        with mlflow.start_run(experiment_id=experiment_id):
            build = get_build(mip_focus)
            opt_model = low_sugar.Model(build)
            result = opt_model.optimize(data=data, callback=callback)

            # Note: Above is replacement for opt_model.fit(data, fit_callback) and opt_model.predict(data)
            mlflow.log_param("MIPFocus", result["callback_result"]["MIPFocus"])
            mlflow.log_metric("RunTime", result["callback_result"]["RunTime"])
            mlflow.log_metric("color_count", result["callback_result"]["color_count"])

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            print(f"tracking_url_type_store: {tracking_url_type_store}")
