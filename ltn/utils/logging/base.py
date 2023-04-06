from __future__ import annotations

from abc import ABC, abstractmethod


class MetricsLogger(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.step = 0

    @abstractmethod
    def log_value(self, name: str, value: float, step: int = None) -> None:
        pass

    @abstractmethod
    def log_dict_of_values(
        self, names_to_values: dict[str, float], step: int = None
    ) -> None:
        pass

    def increment_step(self) -> None:
        self.step += 1
