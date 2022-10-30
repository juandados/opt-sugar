from itertools import product
import numpy as np
from random import random
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from collections import defaultdict
import gurobipy as gp


def generate_graph_data(node_count=16, edge_probability=0.5) -> object:
    nodes = set(range(node_count))
    edges = set(
        (v1, v2)
        for v1, v2 in product(nodes, nodes)
        if random() <= edge_probability and v1 > v2
    )
    edges.update([(1, 0)])  # Forcing the edge 1, 0 to avoid empty edges
    data = {"nodes": nodes, "edges": edges}
    return data


def get_graph_to_show(data, solution):
    cmap = plt.get_cmap("gist_rainbow")
    nodes = data['nodes']
    edges = data['edges']
    color_count = int(solution["max_color"] + 1)

    color_map = {
        node: color_label
        for node, color_label in product(nodes, range(color_count))
        if solution["color"][node, color_label] == 1
    }
    rgb_colors = {
        c: cmap(x) for c, x in enumerate(np.linspace(0, 1 - 1 / color_count, color_count))
    }
    g = Network(notebook=True)
    g.add_nodes(
        nodes=list(color_map.keys()),
        color=[rgb2hex(rgb_colors[color_label]) for color_label in color_map.values()],
    )
    g.add_edges(edges)
    g.set_options(
        """{"edges": {"color": {"inherit": false}}, "physics":{"maxVelocity": 15}}"""
    )
    return g


class ColoringModelBuilder:

    def __init__(self, data):
        self.data = data
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
        base_model.setObjective(max_color, gp.GRB.MINIMIZE)