import re
from collections import defaultdict
from typing import Callable


class Model:
    def __init__(self, build):
        self.data = None

        def build_():
            return build(self.data)

        self.build = build_

    def fit(self):
        """Defining this to avoid warning from mlflow"""
        return self

    def predict(self, data):
        """Defining this to avoid warning from mlflow"""
        return self.optimize(data)

    def optimize(self, data, callback: Callable = lambda model: dict()):
        """
        :param data:
        :param callback: Check (https://www.gurobi.com/documentation/9.5/refman/attributes.html)
        :return: results
        """
        self.data = data
        model = self.build()
        model.optimize()
        result = self._get_result(model)
        callback_result = callback(model)
        if callback_result:
            result = {**result, "callback_result": callback_result}
        return result

    def _get_result(self, model):
        vars = defaultdict(dict)
        for v in model.getVars():
            try:
                main_name, index = self.parse_var_name(v.var_name)
                vars[main_name] = {**vars[main_name], index: v.x}
            except TypeError:
                vars[v.var_name] = v.x
        result = {
            "vars": dict(vars),
            "objective_value": model.getObjective().getValue(),
        }
        return result

    @staticmethod
    def parse_var_name(var_name):
        m = re.match(r"(?P<group_name>\w+)\[(?P<index>[\w|\,]+)\]", var_name)
        group_name = m["group_name"]
        index = m["index"]
        index = tuple(int(ind) if ind.isdigit() else ind for ind in index.split(","))
        return group_name, index
