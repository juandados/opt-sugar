from abc import abstractmethod
import json
import grblogtools as glt
from sklearn.utils.validation import check_is_fitted
import gurobipy as gp
from . import objective
import tempfile


class ModelBuilder:
    """This should be user implemented"""

    def __init__(self, data):
        self.data = data
        self.objective = None

    @abstractmethod
    def build_variables(self, base_model: gp.Model) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} should implement build_variables!"
        )

    @abstractmethod
    def build_constraints(self, base_model: gp.Model) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} should implement build_constraints!"
        )

    @abstractmethod
    def build_objective(self, base_model) -> objective.Objective:
        raise NotImplementedError(
            f"{type(self).__name__} should implement build_objective!"
        )

    def build(self, name="my_model"):
        base_model = gp.Model(name)
        self.build_variables(base_model)
        base_model.update()
        self.build_constraints(base_model)
        base_model.update()
        self.objective = self.build_objective(base_model)
        base_model.update()
        return base_model


def model_builder_factory(variables_builder, constraints_builder, objective_builder):
    class ModelBuilderF(ModelBuilder):
        def build_variables(self, base_model: gp.Model) -> None:
            variables_builder(base_model, self.data)

        def build_constraints(self, base_model: gp.Model) -> None:
            constraints_builder(base_model, self.data)

        def build_objective(self, base_model: gp.Model) -> objective.Objective:
            return objective_builder(base_model, self.data)

    return ModelBuilderF


class OptModel:
    vars_: dict
    objective_value_: float
    fit_callback_data: dict

    def __init__(self, *, model_builder):
        self.model_builder = model_builder
        self.data = None
        self.objective = None
        self.nodelog_progress = None
        self.log_results = None
        self.model = None

    def fit(self, data, callback=None, log_file=None):
        """Builds and optimize the specific the model given the data
        Notice the model is not part of the class, so if we want to read attributes of the model it is needed the callback
        The callback will be executed after model optimization.
        """
        self.data = data
        # TODO: add some checks over data here may be feasibility
        model_builder = self.model_builder(data)
        model = model_builder.build()

        with open(log_file, mode='w+b') if log_file else tempfile.NamedTemporaryFile() as file:
            model.setParam('LogFile', file.name)
            model.optimize()
            self.log_results = glt.parse([file.name])

        self.objective = json.loads(model_builder.objective.__repr__())

        try:
            self.vars_ = {v.var_name: v.x for v in model.getVars()}
            self.objective_value_ = model.getObjective().getValue()
            if callback:
                self.fit_callback_data = callback(model)
        except Exception:
            del self.vars_, self.objective_value_, self.fit_callback_data
        finally:
            del model
        return self

    def predict(self, data, *args, **kwargs):
        """Fits estimator if not fitted or self.data differs from data and returns the
        variable values"""
        fitted = [v for v in vars(self) if v.endswith("_") and not v.startswith("__")]
        if not fitted or self.data != data:
            self.fit(data, *args, **kwargs)
        return self.vars_

    def optimize(self, data, *args, **kwargs):
        return self.fit(data, *args, **kwargs)

    def score(self):
        """Returns the spefic model objective given the data"""
        check_is_fitted(self)
        return self.objective_value_
