from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import MultiOutputMixin
from sklearn.utils.validation import check_is_fitted
from abc import abstractmethod
import gurobipy as gp
from .objective import Objective, ObjectivePart, BaseObjective
from copy import copy
import json


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
    def build_objective(self, base_model) -> Objective:
        raise NotImplementedError(
            f"{type(self).__name__} should implement build_objective!"
        )

    def build(self):
        base_model = gp.Model("my_model")
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

        def build_objective(self, base_model: gp.Model) -> Objective:
            return objective_builder(base_model, self.data)

    return ModelBuilderF


class OptModel:  # (MultiOutputMixin, RegressorMixin, BaseEstimator):
    vars_: dict
    objective_value_: float
    fit_callback_data: dict

    def __init__(self, *, model_builder):
        self.ModelBuilder = model_builder
        self.data = None
        self.objective = None

    def fit(self, data, callback=None):
        self.data = data
        """Builds and optimize the specific the model given the data"""
        # TODO: add some checks over data here may be feasibility
        model_builder = self.ModelBuilder(data)
        model = model_builder.build()
        model.optimize()

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

    def predict(self, data, callback=None):
        """Fits estimator if not fitted or self.data differs from data and returns the variable values"""
        fitted = [v for v in vars(self) if v.endswith("_") and not v.startswith("__")]
        if not fitted or self.data != data:
            self.fit(data, callback=callback)
        return self.vars_

    def optimize(self, data, callback=None):
        return self.predict(data, callback=callback)

    def score(self):
        """Returns the spefic model objective given the data"""
        check_is_fitted(self)
        return self.objective_value_
