from collections import defaultdict
from random import random, seed
from itertools import product
import pytest
import gurobipy as gp
from src.opt_sugar.extra_sugar import (
    OptModel,
    ModelBuilder,
    ObjectivePart,
    Objective,
    BaseObjective,
)


class ColoringModelBuilder(ModelBuilder):
    """This should be user implemented"""

    def __init__(self, data):
        super().__init__(data)
        self.degree = None
        self.variables = None

    def build_variables(self, base_model):
        degrees = defaultdict(int)
        for node1, node2 in self.data["edges"]:
            degrees[node1] += 1
            degrees[node2] += 1
        self.degree = max(degrees.items(), key=lambda x: x[1])[1]
        color_keys = list(product(self.data["nodes"], range(self.degree)))
        color = base_model.addVars(color_keys, vtype="B", name="color")
        max_color = base_model.addVar(lb=0, ub=self.degree, vtype="C", name="max_color")
        self.variables = {"color": color, "max_color": max_color}

    def build_constraints(self, base_model):
        color = self.variables["color"]
        for node1, col in color:
            # if color[v1, col] == 1 -> color[v2, col] == 0 for all v2 such that (v1, v2)
            # or belongs to E
            for node2 in self.data["nodes"]:
                if (node2, node1) in self.data["edges"] or (node1, node2) in self.data[
                    "edges"
                ]:
                    base_model.addConstr(
                        color[node2, col] <= 1 - color[node1, col],
                        name=f"color_{col}_{node1}_{node2}",
                    )

        for node in self.data["nodes"]:
            base_model.addConstr(
                gp.quicksum(color[node, col] for col in range(self.degree)) == 1,
                name=f"every_node_has_color_{node}",
            )

        max_color = self.variables["max_color"]
        for node, col in color:
            base_model.addConstr(
                col * color[node, col] <= max_color, name=f"max_color_{node}_{col}"
            )

    def build_objective(self, base_model):
        max_color = self.variables["max_color"]
        objective_parts = [ObjectivePart(weight=1, expr=max_color)]
        objective = Objective([BaseObjective(objective_parts, hierarchy=1)])
        base_model.setObjective(objective.build()[0], gp.GRB.MINIMIZE)
        return objective


@pytest.fixture
def five_node_data():
    node_count = 5
    nodes = set(range(node_count))
    edge_probability = 0.5
    seed(10)
    edges = set(
        (v1, v2)
        for v1, v2 in product(nodes, nodes)
        if random() <= edge_probability and v1 > v2
    )
    edges.update([(1, 0)])  # Forcing the edge 1, 0 to avoid empty edges
    data = {"nodes": nodes, "edges": edges}
    return data


# pylint: disable=no-self-use, redefined-outer-name
@pytest.mark.unit
class TestOptModel:
    def test_fit(self, five_node_data):
        opt_model = OptModel(model_builder=ColoringModelBuilder)
        opt_model.fit(five_node_data)
        # color count is 2
        color_count = opt_model.objective_value_ + 1
        assert color_count == 2

    def test_predict(self, five_node_data):
        opt_model = OptModel(model_builder=ColoringModelBuilder)
        vars_ = opt_model.predict(five_node_data)
        assert isinstance(vars_, dict)
        # color count is 2
        color_count = opt_model.objective_value_ + 1
        assert color_count == 2
