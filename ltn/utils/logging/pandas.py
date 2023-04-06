from __future__ import annotations

import pandas as pd

import ltn.utils.logging.base as base


class DataFrameLogger(base.MetricsLogger):
    def __init__(self) -> None:
        super().__init__()
        self.df = pd.DataFrame()

    def log_value(self, name: str, value: float, step: int = None) -> None:
        if name == "step":
            raise ValueError("Metrics name cannot be 'step'.")
        step = step if step else self.step
        self.df.loc[step, name] = value

    def log_dict_of_values(
        self, names_to_values: dict[str, float], step: int = None
    ) -> None:
        step = step if step else self.step
        for name, value in names_to_values.items():
            self.log_value(name, value, step=step)

    def get_df(self) -> pd.DataFrame:
        df = self.df.sort_index()
        df["step"] = df.index
        df = df.reset_index()
        return df

    def to_csv(self, path: str) -> None:
        df = self.get_df()
        df.to_csv(path, index=False)
