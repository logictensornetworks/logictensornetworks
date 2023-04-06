from __future__ import annotations
from typing import Callable, Any
import dataclasses
import itertools

import tensorflow as tf

import ltn


@dataclasses.dataclass  
class Grounding:
    constants: dict[str, ltn.Constant] = dataclasses.field(default_factory=dict)
    variables: dict[str, ltn.Variable] = dataclasses.field(default_factory=dict)
    functions: dict[str, ltn.Function] = dataclasses.field(default_factory=dict)
    predicates: dict[str, ltn.Predicate] = dataclasses.field(default_factory=dict)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return list(itertools.chain(*[x.trainable_variables for x in itertools.chain(self.predicates.values(), 
                self.functions.values(), self.constants.values())]))


@dataclasses.dataclass
class OperatorConfig:
    not_: ltn.Wrapper_Connective 
    and_: ltn.Wrapper_Connective
    or_: ltn.Wrapper_Connective
    implies: ltn.Wrapper_Connective
    exists: ltn.Wrapper_Quantifier
    forall: ltn.Wrapper_Quantifier
    and_aggreg: ltn.Wrapper_Formula_Aggregator = None
    or_aggreg: ltn.Wrapper_Formula_Aggregator = None
    schedules: list[_OperatorSchedule] = dataclasses.field(default_factory=list)

    def set_schedule(
            self, 
            operator: ltn.Wrapper_Connective | ltn.Wrapper_Formula_Aggregator | ltn.Wrapper_Quantifier, 
            param_key: str, 
            schedule: dict[Any, float],
            ) -> None:
        self.schedules.append(_OperatorSchedule(operator, param_key=param_key, schedule=schedule))

    def update_schedule(
            self,
            schedule_key: Any
            ) -> None:
        for schedule in self.schedules:
            schedule.update_schedule(schedule_key)


@dataclasses.dataclass
class _OperatorSchedule:
    operator : ltn.Wrapper_Connective | ltn.Wrapper_Quantifier | ltn.Wrapper_Formula_Aggregator
    param_key : str
    schedule: dict[Any, float]
    operator_callable : Callable = None

    def __post_init__(self) -> None:
        if isinstance(self.operator, ltn.Wrapper_Connective):
            self.operator_callable = self.operator.connective_op
        elif isinstance(self.operator, (ltn.Wrapper_Quantifier, ltn.Wrapper_Formula_Aggregator)):
            self.operator_callable = self.operator.aggreg_op
        else:
            raise ValueError("`operator_wrapper` argument must be of type `ltn.Wrapper_Connective`,"\
                    "`ltn.Wrapper_Formula_Aggregator`, `ltn.Wrapper_Quantifier`.")        
        if not hasattr(self.operator_callable, self.param_key):
            raise ValueError(f"Operator {self.operator_callable} doesn't have the parameter attribute "\
                    f"{self.param_key}.")

    def update_schedule(self, schedule_key: Any) -> None:
        setattr(self.operator_callable, self.param_key, self.schedule[schedule_key])