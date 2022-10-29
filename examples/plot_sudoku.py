"""
========================================
Low Sugar: Solving Sudokus Like a Master
========================================

This example demostrates how to use the low-sugar in combination with mlflow
to solve sudoku puzzles.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_sudoku.ipynb
"""

import numpy as np
import sys
import mlflow
from mlflow import MlflowException
from itertools import product
import datetime
import pathlib
import pandas as pd

import gurobipy as gp

sys.path.append("/Users/Juan.ChaconLeon/opt/opt-sugar/src")  # when running locally
from opt_sugar import low_sugar
from opt_sugar import opt_flow
from utils.sudoku import show_sudoku


experiment_name = f"sudoku_{datetime.datetime.now().strftime('%Y_%m_%d')}"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):

    def build(data):
        # Create a new model
        m = gp.Model("sudoku")

        # Create Indices
        pos_ys = list(range(3))
        pos_xs = list(range(3))
        square_ys = list(range(3))
        square_xs = list(range(3))
        positions = list(product(pos_ys, pos_xs, square_ys, square_xs))

        digits = list(range(1, 10))
        indices = list((*pos, digit) for pos, digit in product(positions, digits))

        # Create variables
        digit_pick = m.addVars(indices, vtype="B", name="digit_pick")

        # Set objective: This is a feasibility problem rather than on optimization one.
        m.setObjective(0)

        # Add constraints:
        for pos_y, square_y, digit in product(pos_ys, square_ys, digits):
            m.addConstr(
                digit_pick.sum(pos_y, "*", square_y, "*", digit) == 1,
                name=f"row_{pos_y}_{square_y}_{digit}",
            )

        for pos_x, square_x, digit in product(pos_xs, square_xs, digits):
            m.addConstr(
                digit_pick.sum("*", pos_x, "*", square_x, digit) == 1,
                name=f"col_{pos_x}_{square_x}_{digit}",
            )

        for square_y, square_x, digit in product(square_ys, square_xs, digits):
            m.addConstr(
                digit_pick.sum("*", "*", square_y, square_x, digit) == 1,
                name=f"square_{square_y}_{square_x}_{digit}",
            )

        for pos_y, pos_x, square_y, square_x in positions:
            y = square_y * 3 + pos_y
            x = square_x * 3 + pos_x
            chosen_digit = data[y][x]
            if chosen_digit:
                chosen_digit = int(chosen_digit)
                m.addConstr(
                    digit_pick[pos_y, pos_x, square_y, square_x, chosen_digit] == 1,
                    name=f"chosen_{pos_y}_{pos_x}_{square_y}_{square_x}_{chosen_digit}",
                )

        for pos_y, pos_x, square_y, square_x in positions:
            m.addConstr(
                digit_pick.sum(pos_y, pos_x, square_y, square_x, "*") == 1,
                name=f"position_{pos_y}_{pos_x}_{square_y}_{square_x}_chosen",
            )

        return m

    base_path = pathlib.Path(".").absolute()
    examples_path = base_path / "data"
    data = (
        pd.read_csv(examples_path / "sudoku_1.csv", header=None)
        .replace({np.NaN: None})
        .values
    )

    # Building
    opt_model = low_sugar.Model(build)
    result = opt_model.optimize(data=data)
    show_sudoku(vars=result["vars"]["digit_pick"])
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

base_path = pathlib.Path(".").absolute()
examples_path = base_path / "data"
data = (
    pd.read_csv(examples_path / "sudoku_2.csv", header=None)
    .replace({np.NaN: None})
    .values
)
result = loaded_model.optimize(data=data)

# Using a util function (Check imports)
show_sudoku(vars=result["vars"]["digit_pick"])
print(f"solution from the registered model {result['objective_value']}")

# %%
# .. image:: https://mybinder.org/badge_logo.svg
#  :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_sudoku.ipynb
