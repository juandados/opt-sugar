import re
from collections import defaultdict


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

    def optimize(self, data):
        self.data = data
        model = self.build()
        model.optimize()
        vars = defaultdict(dict)
        for v in model.getVars():
            main_name, index = self.parse_var_name(v.var_name)
            vars[main_name] = {**vars[main_name], index: v.x}
        result = {
            "vars": dict(vars),
            "objective_value": model.getObjective().getValue()
        }
        return result

    @staticmethod
    def parse_var_name(var_name):
        m = re.match(r"(?P<group_name>\w+)\[(?P<index>[\w|\,]+)\]", var_name)
        group_name = m['group_name']
        index = m['index']
        index = tuple(int(ind) if ind.isdigit() else ind for ind in index.split(','))
        return group_name, index

