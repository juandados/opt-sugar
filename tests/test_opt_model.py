import pytest
from itertools import product
from random import random, seed
from src.opt_sugar import (
    OptModel,
    ModelBuilder,
    ObjectivePart,
    Objective,
    BaseObjective,
)
from collections import defaultdict
import gurobipy as gp


class ColoringModelBuilder(ModelBuilder):
    """This should be user implemented"""

    def __init__(self, data):
        super().__init__(data)
        self.degree = None
        self.variables = None

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
            # if color[v1, c] == 1 -> color[v2, c] == 0 for all v2 such that (v1, v2) or  belongs to E
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


@pytest.fixture
def five_vertex_data():
    vertex_count = 5
    vertex = set(range(vertex_count))
    edge_probability = 0.5
    seed(10)
    edges = set(
        (v1, v2)
        for v1, v2 in product(vertex, vertex)
        if random() <= edge_probability and v1 > v2
    )
    edges.update([(1, 0)])  # Forcing the edge 1, 0 to avoid empty edges
    data = {"nodes": vertex, "edges": edges}
    return data


@pytest.mark.unit
class TestOptModel:
    def test_fit(self, five_vertex_data):
        opt_model = OptModel(model_builder=ColoringModelBuilder)
        opt_model.fit(five_vertex_data)
        # color count is 2
        color_count = opt_model.objective_value_ + 1
        assert color_count == 2

    def test_predict(self, five_vertex_data):
        opt_model = OptModel(model_builder=ColoringModelBuilder)
        vars_ = opt_model.predict(five_vertex_data)
        assert isinstance(vars_, dict)
        # color count is 2
        color_count = opt_model.objective_value_ + 1
        assert color_count == 2
