from __future__ import annotations

import tensorflow as tf

import ltn.utils.logging.base as base


class TfSummaryLogger(base.MetricsLogger):
    def __init__(self, summary_writer: tf.summary.SummaryWriter) -> None:
        super().__init__()
        self.summary_writer = summary_writer

    def log_value(self, name: str, value: float, step: int = None) -> None:
        step = step if step else self.step
        with self.summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def log_dict_of_values(
        self, names_to_values: dict[str, float], step: int = None
    ) -> None:
        step = step if step else self.step
        for name, value in names_to_values.items():
            self.log_value(name, value, step=step)
