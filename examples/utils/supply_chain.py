import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
from collections import defaultdict


def plot_supply_chain(data, solution):
    days = sorted(int(x) for x in set(solution["dispatch"].values()) | {0})
    components = list(data["initial_inventory"].keys())
    customers = list(data["demand"].keys())
    products = list(data["recipe"])
    customer_products = list(product(customers, products))
    network = dict()
    labels = defaultdict(dict)
    edge_labels = defaultdict(dict)

    max_y = 10
    pos = dict()
    pos.update({"factory": (0, max_y / (2 + 1))})
    pos.update({"extra": (0, 2 * max_y / (2 + 1))})
    pos.update(
        {
            (comp, "inv"): (1, i * max_y / (len(components) - 1))
            for i, comp in enumerate(components)
        }
    )
    pos.update(
        {
            (customer, prod): (2, i * max_y / (len(customer_products) - 1))
            for i, (customer, prod) in enumerate(customer_products)
        }
    )

    nodes = [
        "factory",
        "extra",
        *[(comp, "inv") for comp in components],
        *customer_products,
    ]

    for t in days:
        network[t] = nx.DiGraph()
        network[t].add_nodes_from(nodes)

        for (comp, t_), val in solution["inventory"].items():
            if t != t_:
                continue
            if val != 0:
                labels[t][(comp, "inv")] = f"{comp.title()}\n Inv:\n {val}"

        labels[t].update(
            {
                (customer, prod): f"{customer.title()}\n{prod.title()}"
                for customer, prod in customer_products
            }
        )
        labels[t].update({"factory": "Factory"})
        labels[t].update({"extra": "Extra\nProduction"})

        for (customer, prod, comp, t_), val in solution["from_inventory"].items():
            if t != t_:
                continue
            if val != 0:
                network[t].add_edge((comp, "inv"), (customer, prod))
                edge_labels[t][((comp, "inv"), (customer, prod))] = val

        for (customer, prod, comp, t_), val in solution["from_factory"].items():
            if t != t_:
                continue
            if val != 0:
                network[t].add_edge("factory", (customer, prod))
                edge_labels[t][("factory", (customer, prod))] = f"{comp}: {val}"

        for (comp, t_), val in solution["to_inventory"].items():
            if t != t_:
                continue
            if val != 0:
                network[t].add_edge("factory", (comp, "inv"))
                edge_labels[t][("factory", (comp, "inv"))] = f"{comp}: {val}"

        for (customer, prod, comp, t_), val in solution["extra_production"].items():
            if t != t_:
                continue
            if val != 0:
                network[t].add_edge("extra", (customer, prod))
                edge_labels[t][("extra", (customer, prod))] = f"{comp}: {val}"

        plt.figure()
        nx.draw(
            network[t],
            with_labels=True,
            pos=pos,
            labels=labels[t],
            edge_color="black",
            width=1,
            linewidths=1,
            node_size=4000,
            node_color="pink",
            alpha=0.9,
        )
        nx.draw_networkx_edge_labels(
            network[t], pos=pos, edge_labels=edge_labels[t], font_color="red"
        )