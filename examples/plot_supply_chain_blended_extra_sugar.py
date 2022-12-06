"""
=============================
Experiment Tracking: Coloring
=============================

This example demostrates how to use opt-sugar in combination with mlflow
for single objective optimization experiment tracking

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/juandados/opt-sugar/main?labpath=doc%2Fsource%2Fauto_examples%2Fplot_coloring.ipynb
"""
from itertools import product
import logging
import json
import pathlib
import datetime

import mlflow
from mlflow.exceptions import MlflowException

import gurobipy as gp
import sys; sys.path.append('/Users/Juan.ChaconLeon/opt/opt-sugar/src')  # when running locally
from opt_sugar.extra_sugar import OptModel, ModelBuilder
from opt_sugar.extra_sugar.objective import Objective, ObjectivePart, BaseObjective


class SupplyChainBlendedModelBuilder(ModelBuilder):
    """This should be user implemented"""

    def __init__(self, data):
        super().__init__(data)
        self.variables = None
        self.indices = self._build_indices()

    def _build_indices(self):
        customers = list(self.data["demand"].keys())
        accessories = list(self.data["initial_inventory"].keys())
        products = list(self.data["recipe"].keys())
        max_day = max(
            demand_details["date"] for demand_details in self.data["demand"].values()
        )
        days = range(1, max_day + 1)
        customer_dates = {
            (customer, day)
            for customer in customers
            for day in range(
                self.data["demand"][customer]["date"],
                min(
                    self.data["demand"][customer]["date"] + self.data["max_delay"],
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

    def build_variables(self, base_model):
        indices = self.indices
        dispatch = base_model.addVars(
            indices["customer_dates"], vtype="B", name="dispatch"
        )
        inventory = base_model.addVars(
            product(indices["accessories"], indices["days"]),
            vtype="C",
            name="inventory",
        )
        from_inventory = base_model.addVars(
            product(
                indices["customers"],
                indices["products"],
                indices["accessories"],
                indices["days"],
            ),
            vtype="I",
            name="from_inventory",
        )
        from_factory = base_model.addVars(
            product(
                indices["customers"],
                indices["products"],
                indices["accessories"],
                indices["days"],
            ),
            vtype="I",
            name="from_factory",
        )
        to_inventory = base_model.addVars(
            product(indices["accessories"], indices["days"]),
            vtype="C",
            name="to_inventory",
        )

        extra_production = base_model.addVars(
            product(
                indices["customers"],
                indices["products"],
                indices["accessories"],
                indices["days"],
            ),
            vtype="C",
            name="extra_production",
        )

        self.variables = {
            "dispatch": dispatch,
            "inventory": inventory,
            "from_inventory": from_inventory,
            "from_factory": from_factory,
            "to_inventory": to_inventory,
            "extra_production": extra_production,
        }
        return self.variables

    def build_constraints(self, base_model):
        accessories = self.indices["accessories"]
        days = self.indices["days"]
        customers = self.indices["customers"]
        products = self.indices["products"]
        customer_dates = self.indices["customer_dates"]

        to_inventory = self.variables["to_inventory"]
        from_factory = self.variables["from_factory"]
        from_inventory = self.variables["from_inventory"]
        inventory = self.variables["inventory"]
        dispatch = self.variables["dispatch"]
        extra_production = self.variables["extra_production"]

        production = self.data["production"]
        initial_inventory = self.data["initial_inventory"]
        inventory_capacity = self.data["inventory_capacity"]
        demand = self.data["demand"]
        recipe = self.data["recipe"]

        for accessory, day in product(accessories, days):
            if str(day) in production[accessory]:
                base_model.addConstr(
                    to_inventory[accessory, day]
                    + from_factory.sum("*", "*", accessory, day)
                    == production[accessory][str(day)],
                    name=f"production_allocation_{accessory}_{day}",
                )
            else:
                base_model.addConstr(
                    to_inventory[accessory, day]
                    + from_factory.sum("*", "*", accessory, day)
                    == 0,
                    name=f"production_allocation_{accessory}_{day}",
                )

        for accessory, day in product(accessories, days):
            if day == 1:
                base_model.addConstr(
                    inventory[accessory, day]
                    == initial_inventory[accessory] + to_inventory[accessory, day] - from_inventory.sum("*", "*", accessory, day),
                    name=f"keeping_track_of_inventories_{accessory}_{day}",
                )
            else:
                base_model.addConstr(
                    inventory[accessory, day]
                    == inventory[accessory, day - 1] + to_inventory[accessory, day] - from_inventory.sum("*", "*", accessory, day),
                    name=f"keeping_track_of_inventories_{accessory}_{day}",
                )

        for accessory, day in product(accessories, days):
            base_model.addConstr(
                inventory[accessory, day] <= inventory_capacity[accessory],
                name=f"inventory_capacity_{accessory}_{day}",
            )

        for customer in customers:
            base_model.addConstr(
                dispatch.sum(customer, "*") == 1, name=f"customer_served_{customer}"
            )

        for prod, accessory in product(products, accessories):
            for customer, day in customer_dates:
                base_model.addConstr(
                    from_inventory[customer, prod, accessory, day]
                    + from_factory[customer, prod, accessory, day]
                    + extra_production[customer, prod, accessory, day]
                    == dispatch[customer, day]
                    * demand[customer][prod]
                    * recipe[prod][accessory],
                    name=f"demand_satisfaction_{customer}_{prod}_{accessory}_{day}",
                )

    def build_objective(self, base_model):
        dispatch = self.variables["dispatch"]
        extra_production = self.variables["extra_production"]
        customers = self.indices["customers"]
        demand = self.data["demand"]

        delay_penalty_costs = gp.quicksum(
            gp.quicksum(
                day * dispatch[customer, day]
                for customer, day in dispatch
                if customers == customer_
            )
            - demand[customer_]["date"]
            for customer_ in customers
        )
        extra_production_costs = 2 * extra_production.sum()

        objective_parts = [
            ObjectivePart(weight=0.5, expr=delay_penalty_costs),
            ObjectivePart(weight=0.5, expr=extra_production_costs),
        ]

        objective = Objective([BaseObjective(objective_parts, hierarchy=1)])
        base_model.setObjective(objective.build()[0], gp.GRB.MINIMIZE)
        return objective


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


base_path = pathlib.Path('.')
examples_path = base_path.parent / "data"

with open(examples_path / "supply_chain_blended_mini_toy.json", "r", encoding="utf-8") as file:
    data = json.load(file)

with mlflow.start_run(experiment_id=experiment_id):
    opt_model = OptModel(model_builder=SupplyChainBlendedModelBuilder)
    solution = opt_model.optimize(data)
