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

        vars = {v.var_name: v.x for v in model.getVars()}
        objective_value = model.getObjective().getValue()

        return vars, objective_value

