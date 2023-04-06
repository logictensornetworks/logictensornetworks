from __future__ import annotations
from typing import Callable, Any
import abc

import numpy as np

from ltn.wrapper.grounding import Grounding, OperatorConfig
from ltn.wrapper.domains import Domain
import ltn

class Constraint(abc.ABC):
    def __init__(
            self, 
            label: str, 
            grounding: Grounding, 
            operator_config: OperatorConfig,
            doms_feed_dict: dict[str, Domain] = None
    ) -> None:
        super().__init__()
        self.label = label
        self.grounding = grounding
        self.operator_config = operator_config
        self.doms_feed_dict = doms_feed_dict
        
    @abc.abstractmethod
    def formula(self):
        pass

    def call_with_domains(
            self,
            **kwargs
    ) -> ltn.Formula:
        if self.doms_feed_dict is None:
            raise ValueError("There is no domains associated with this constraint yet. "
                    "The constraint cannot pull values directly from domains.")
        feed_dict = {key: dom.current_minibatch for (key, dom) in self.doms_feed_dict.items()}
        return self.call_with_feed_dict(feed_dict, **kwargs)
    
    def call_with_feed_dict(
            self,
            feed_dict = dict[str, np.ndarray],
            **kwargs
    ) -> ltn.Formula:
        return compute_formula_with_feed_dict(formula_fn=self.formula, feed_dict=feed_dict,
                formula_kwargs=kwargs)


def compute_formula_with_feed_dict(
        formula_fn: Callable, 
        feed_dict: dict[str, np.ndarray],
        formula_kwargs: dict[str, Any] = None
        ) -> ltn.Formula:
    """Compute a ltn formula for some tuples of variable values, in order (diagged).

    Args:
        formula_fn (Callable): A function that accepts `ltn.Variable` in keyword arguments and 
                returns an `ltn.Formula`.
        feed_dict (dict[str, np.ndarray]): the variable values, in a dictionary where the keys are
                the keywords used in `formula_fn` for each variable. The values are arrays of the
                same length that will be diagged. 
        formula_kwargs (dict[str, Any], optional): Other kwargs for the formula_fn. 
                Defaults to None.

    Returns:
        ltn.Formula: Result of the computation.
    """
    # create ltn variables and diag lock the tuples
    diagged_vars = {label: ltn.Variable(label, values) for label, values in feed_dict.items()}
    ltn.diag_lock(*diagged_vars.values())
    # compute sat level
    formula_kwargs = {} if not formula_kwargs else formula_kwargs
    return formula_fn(**diagged_vars, **formula_kwargs)
