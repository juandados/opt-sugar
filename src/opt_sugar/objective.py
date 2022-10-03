from typing import Union, List, Iterable
import json


def pretty_json_str(json_str):
    return json.dumps(json.loads(json_str.replace("'", '"')), indent=2, sort_keys=True)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class ObjectivePart:
    def __init__(self, weight: float, expr):
        self._weight = weight
        self._expr = expr

    def __add__(self, other):
        expr = self._weight * self._expr + other._weight * other._expr
        return ObjectivePart(weight=1, expr=expr)

    def __repr__(self):
        return json.dumps(
            {k: v if is_jsonable(v) else str(v) for k, v in self.__dict__.items()},
            indent=2,
            sort_keys=True,
        )


class BaseObjective:
    def __init__(self, objective_parts: List[ObjectivePart], hierarchy: int):
        self._objective_parts = objective_parts
        self._hierarchy = hierarchy

    def build(self):
        objective_parts_sum = sum(
            (objective_part for objective_part in self._objective_parts),
            ObjectivePart(weight=1, expr=0),
        )
        return objective_parts_sum._expr

    def __repr__(self):
        return pretty_json_str(str(self.__dict__))


class Objective:
    def __init__(self, base_objectives: Iterable[BaseObjective]):
        self._base_objectives = base_objectives

    def build(self) -> Union[BaseObjective, List[BaseObjective]]:
        try:
            return [base_objective.build() for base_objective in self._base_objectives]
        except TypeError:
            return self._base_objectives.build()

    def __repr__(self):
        return pretty_json_str(str(self.__dict__))
