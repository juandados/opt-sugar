"""
========================================
Superheroes Factory
========================================

This example demostrates how to use the low-sugar in combination with mlflow
to solve a simple supply chain problem.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_supply_chain_blended.ipynb
"""
# sphinx_gallery_thumbnail_path = '_static/superheroes.png'

# %%
# Problem Description
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Populate description here later.

# %%
# Let's start by doing some useful imports.

from itertools import product
import gurobipy as gp
from typing import Dict


# %%
# Tracking an Optimization Experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.
def build_indices(data: Dict) -> Dict:
    customers = list(data["demand"].keys())
    accessories = list(data["initial_inventory"].keys())
    products = list(data["recipe"].keys())
    max_day = max(demand_details["date"] for demand_details in data["demand"].values())
    days = range(1, max_day + 1)
    customer_dates = {
        (customer, day)
        for customer in customers
        for day in range(
            data["demand"][customer]["date"],
            min(
                data["demand"][customer]["date"] + data["max_delay"],
                max_day + 1,
            ),
        )
    }

    indices = {
        "customers": customers,
        "customer_dates": customer_dates,
        "max_day": max_day,
        "days": days,
        "accessories": accessories,
        "products": products,
    }
    return indices


def build_variables(model: gp.Model, indices: Dict):
    dispatch = model.addVars(indices["customer_dates"], vtype="B", name="dispatch")
    inventory = model.addVars(
        product(indices["accessories"], indices["days"]),
        vtype="C",
        name="inventory",
    )
    from_inventory = model.addVars(
        product(
            indices["customers"],
            indices["products"],
            indices["accessories"],
            indices["days"],
        ),
        vtype="I",
        name="from_inventory",
    )
    from_factory = model.addVars(
        product(
            indices["customers"],
            indices["products"],
            indices["accessories"],
            indices["days"],
        ),
        vtype="I",
        name="from_factory",
    )
    to_inventory = model.addVars(
        product(indices["accessories"], indices["days"]),
        vtype="C",
        name="to_inventory",
    )

    extra_production = model.addVars(
        product(
            indices["customers"],
            indices["products"],
            indices["accessories"],
            indices["days"],
        ),
        vtype="C",
        name="extra_production",
    )

    variables = {
        "dispatch": dispatch,
        "inventory": inventory,
        "from_inventory": from_inventory,
        "from_factory": from_factory,
        "to_inventory": to_inventory,
        "extra_production": extra_production,
    }
    return variables


def build_constraints(model: gp.Model, indices: Dict, variables: Dict) -> gp.Model:
    to_inventory = variables["to_inventory"]
    from_factory = variables["from_factory"]
    from_inventory = variables["from_inventory"]
    inventory = variables["inventory"]
    dispatch = variables["dispatch"]
    extra_production = variables["extra_production"]

    production = data["production"]
    initial_inventory = data["initial_inventory"]
    inventory_capacity = data["inventory_capacity"]
    demand = data["demand"]
    recipe = data["recipe"]

    for accessory, day in product(indices["accessories"], indices["days"]):
        if str(day) in production[accessory]:
            model.addConstr(
                to_inventory[accessory, day]
                + from_factory.sum("*", "*", accessory, day)
                == production[accessory][str(day)],
                name=f"production_allocation_{accessory}_{day}",
            )
        else:
            model.addConstr(
                to_inventory[accessory, day]
                + from_factory.sum("*", "*", accessory, day)
                == 0,
                name=f"production_allocation_{accessory}_{day}",
            )

    for accessory, day in product(indices["accessories"], indices["days"]):
        if day == 1:
            model.addConstr(
                inventory[accessory, day]
                == initial_inventory[accessory]
                + to_inventory[accessory, day]
                - from_inventory.sum("*", "*", accessory, day),
                name=f"keeping_track_of_inventories_{accessory}_{day}",
            )
        else:
            model.addConstr(
                inventory[accessory, day]
                == inventory[accessory, day - 1]
                + to_inventory[accessory, day]
                - from_inventory.sum("*", "*", accessory, day),
                name=f"keeping_track_of_inventories_{accessory}_{day}",
            )

    for accessory, day in product(indices["accessories"], indices["days"]):
        model.addConstr(
            inventory[accessory, day] <= inventory_capacity[accessory],
            name=f"inventory_capacity_{accessory}_{day}",
        )

    for customer in indices["customers"]:
        model.addConstr(
            dispatch.sum(customer, "*") == 1, name=f"customer_served_{customer}"
        )

    for prod, accessory in product(indices["products"], indices["accessories"]):
        for customer, day in indices["customer_dates"]:
            model.addConstr(
                from_inventory[customer, prod, accessory, day]
                + from_factory[customer, prod, accessory, day]
                + extra_production[customer, prod, accessory, day]
                == dispatch[customer, day]
                * demand[customer][prod]
                * recipe[prod][accessory],
                name=f"demand_satisfaction_{customer}_{prod}_{accessory}_{day}",
            )

    return model


def build_objective(
    model: gp.Model, indices: Dict, variables: Dict, demand: Dict
) -> gp.Model:
    dispatch = variables["dispatch"]
    extra_production = variables["extra_production"]

    delay_penalty_costs = gp.quicksum(
        gp.quicksum(
            day * dispatch[customer, day]
            for customer, day in dispatch
            if customer == customer_
        )
        - demand[customer_]["date"]
        for customer_ in indices["customers"]
    )

    cost_per_extra_prod = 2
    extra_production_costs = cost_per_extra_prod * extra_production.sum()

    total_cost = delay_penalty_costs + extra_production_costs

    model.setObjective(total_cost, gp.GRB.MINIMIZE)

    return model


def build(data: Dict) -> gp.Model:
    model = gp.Model(name="supply_chain_blended")
    indices = build_indices(data)
    variables = build_variables(model, indices)
    model = build_constraints(model, indices, variables)
    model = build_objective(model, indices, variables, demand=data["demand"])
    model.update()
    return model


# %%
# Tracking an Optimization Experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.
import sys

sys.path.append("/Users/Juan.ChaconLeon/opt/opt-sugar/src")  # when running locally
from opt_sugar import low_sugar

# Setting the experiment
import logging

logging.getLogger("mlflow").setLevel(logging.CRITICAL)  # Can be set DEBUG

import mlflow
from mlflow import MlflowException
import datetime

experiment_name = f"superheros_blended_{datetime.datetime.now().strftime('%Y_%m_%d')}"
try:
    experiment_id = mlflow.create_experiment(name=experiment_name)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

# %%
# Tracking an Optimization Experiment
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add description here.
import requests
import matplotlib.pyplot as plt

with mlflow.start_run(experiment_id=experiment_id):
    examples_path = (
        "https://raw.githubusercontent.com/juandados/opt-sugar/main/examples/data"
    )
    data = requests.get(f"{examples_path}/supply_chain_blended_mini_toy.json").json()

    opt_model = low_sugar.Model(build)
    result = opt_model.optimize(data=data)
    from utils.supply_chain import plot_supply_chain
    plot_supply_chain(data=data, solution=result['vars'])
    plt.show()
    model_info = mlflow.sklearn.log_model(opt_model, "supply_chain_blended")
