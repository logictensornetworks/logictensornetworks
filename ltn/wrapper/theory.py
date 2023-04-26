
from __future__ import annotations
import dataclasses
import tensorflow as tf

import ltn
from ltn.wrapper.constraints import Constraint
from ltn.wrapper.grounding import Grounding
from ltn.wrapper.domains import Domain, DatasetIterator
from ltn.utils.logging.base import MetricsLogger


@dataclasses.dataclass
class Theory:
    constraints: list[Constraint]
    grounding: Grounding
    formula_aggregator: ltn.Wrapper_Formula_Aggregator
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(0.001)
    metrics_loggers: list[MetricsLogger] = dataclasses.field(default_factory=list)
    step: int = 0
    log_every_n_step: int = 50
    agg_sat_metric: tf.keras.metrics.Mean = tf.keras.metrics.Mean("Sat aggregate")
    constraint_metrics: dict[str, tf.keras.metrics.Mean] = dataclasses.field(default_factory=dict)
    
    def __post_init__(self):
        self.constraint_metrics = {constraint.label: tf.keras.metrics.Mean(constraint.label) 
                for constraint in self.constraints}
 
    def train_step_from_domains(
            self,
            constraints_subset: list[Constraint] = None,
            optimizer: tf.keras.optimizers.Optimizer = None
    ) -> None:
        if constraints_subset is not None:
            for constraint in constraints_subset:
                assert(constraint in self.constraints)
        constraints = constraints_subset if constraints_subset is not None else self.constraints
        optimizer = optimizer if optimizer else self.optimizer
        with tf.GradientTape() as tape:
            wffs = [cstr.call_with_domains() for cstr in constraints]
            for (wff, cstr) in zip(wffs, constraints):
                self.constraint_metrics[cstr.label].update_state(wff.tensor)
            agg_sat = self.formula_aggregator(wffs).tensor
            loss = 1-agg_sat
            self.agg_sat_metric.update_state(agg_sat)
        trainable_variables = self.grounding.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.setup_next_minibatches()
        if (self.step % self.log_every_n_step) == 0:
            self.log_metrics()
        self.step += 1

    def log_metrics(self) -> None:
        for metric in self.all_metrics:
            for logger in self.metrics_loggers:
                logger.log_value(metric.name, float(metric.result()), step=self.step)

    def reset_metrics(self) -> None:
        for metric in self.all_metrics:
            metric.reset_state()

    def setup_next_minibatches(self) -> None:
        domains: list[Domain] = []
        for cstr in self.constraints:
            for dom in cstr.doms_feed_dict.values():
                if dom not in domains:
                    domains.append(dom)
        ds_iterators: list[DatasetIterator]  = []
        [ds_iterators.append(dom.dataset_iterator) for dom in domains
                if dom.dataset_iterator not in ds_iterators]
        for ds_iterator in ds_iterators:
            ds_iterator.set_next_minibatch()

    @property
    def all_metrics(self) -> list[tf.keras.metrics.Metric]:
        return [self.agg_sat_metric] + list(self.constraint_metrics.values())