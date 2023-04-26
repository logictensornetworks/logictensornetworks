from __future__ import annotations

from typing import Callable

from ltn.wrapper.grounding import Grounding, OperatorConfig


class Builder:
    def __init__(self, grounding: Grounding, operators: OperatorConfig) -> None:
        self.grounding = grounding
        self.operators = operators

    def build_term(self, parse_results: list) -> Callable:
        pass

