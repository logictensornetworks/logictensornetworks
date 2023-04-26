from __future__ import annotations
from typing import Callable
import dataclasses
import warnings
import copy

import tensorflow as tf
import numpy as np


class Domain:
    """Domain for a `ltn.Variable`. In-memory storage of all the values as a numpy array.
    Also wraps the domain with a `tf.data.Dataset` instance so that constraints can sample 
    minibatches during training.
    """
    def __init__(self, 
            label: str, 
            values: np.ndarray,
            dataset_params: DatasetParams = None
    ) -> None:
        self.label = label
        self.values = values if isinstance(values, np.ndarray) else np.array(values, dtype=np.float32)
        self.dataset_params = dataset_params if dataset_params is not None else DatasetParams()
        self._raw_dataset = tf.data.Dataset.from_tensor_slices(self.values)

        # diagged attributes
        self._diag_label: str = None
        self._diag_dataset: tf.data.Dataset = None
        self._index_in_diag_dataset: int = None
        self._diag_dataset_iterator: DatasetIterator = None

        self.set_dataset_params(dataset_params)
        
    def random_choice_indices(self, size: int, replace: bool) -> np.ndarray:
        return np.random.choice(len(self.values), size=size, replace=replace)

    def random_choice_values(self, size: int, replace: bool, use_map: bool = True) -> np.ndarray:
        values = np.random.choice(self.values, size=size, replace=replace)
        if use_map and self.dataset_params.map_fn is not None:
            values = np.array([self.dataset_params.map_fn(vi) for vi in values])
        return values

    def get_values_at_indices(self, indices: np.ndarray, use_map: bool=True) -> np.ndarray:
        values = tf.gather(self.values, indices, axis=0).numpy()
        if use_map and self.dataset_params.map_fn is not None:
            values = np.array([self.dataset_params.map_fn(vi) for vi in values])
        return values
    
    def set_dataset_params(self, dataset_params: DatasetParams) -> None:
        """In place"""
        if self.is_diagged:
            raise ValueError("Cannot change the settings of a diagged dataset.")
        self.dataset_params = dataset_params
        self._dataset = dataset_params.apply_to_dataset(self._raw_dataset)
        self._dataset_iterator = DatasetIterator(self._dataset)
        
    def _set_diag_dataset(self, diag_dataset: tf.data.Dataset, index_in_diag_dataset: int,
            iterator: DatasetIterator) -> None:
        self._diag_dataset = diag_dataset
        self._index_in_diag_dataset = index_in_diag_dataset
        self._diag_dataset_iterator = iterator

    def undiag(self) -> None:
        self._diag_label = None
        self._diag_dataset = None
        self._index_in_diag_dataset = None
        self._diag_dataset_iterator = None

    @property
    def current_minibatch(self) -> tf.Tensor:
        minibatch = self.dataset_iterator.current_minibatch
        if self.is_diagged:
            minibatch = minibatch[self._index_in_diag_dataset]
        return minibatch

    @property
    def dataset_iterator(self) -> DatasetIterator:
        return self._dataset_iterator if not self.is_diagged else self._diag_dataset_iterator

    @property
    def is_diagged(self) -> bool:
        return self._diag_dataset is not None
    

@dataclasses.dataclass
class DatasetParams:
    shuffle: bool = False
    minibatch_size: int = 32
    drop_remainder: bool = False
    repeat: bool = True
    shuffle_buffer_size: int = None
    map_fn: Callable = None

    def __post_init__(self):
        if self.shuffle and self.shuffle_buffer_size is None:
            raise ValueError("Precise the `shuffle_buffer_size` if using shuffling.")

    def fit(self, other: DatasetParams) -> bool:
        return (self.shuffle == other.shuffle and 
                self.minibatch_size == other.minibatch_size and 
                self.drop_remainder == other.drop_remainder and
                self.repeat == other.repeat and
                self.shuffle_buffer_size == other.shuffle_buffer_size)

    def apply_to_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Not in place"""
        if self.shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        if self.map_fn is not None:
            dataset=dataset.map(self.map_fn)
        dataset = dataset.batch(self.minibatch_size, drop_remainder=self.drop_remainder)
        if self.repeat:
            dataset = dataset.repeat()
        return dataset


@dataclasses.dataclass
class DatasetIterator:
    dataset: tf.data.Dataset
    iterator: tf.data.Iterator = None
    _current_minibatch: tf.Tensor | list[tf.Tensor] = None
    _has_used_current_minibatch: bool = False
    def __post_init__(self):
        self.iterator = iter(self.dataset)
        self._current_minibatch = next(self.iterator)

    @property
    def current_minibatch(self) -> tf.Tensor:
        self._has_used_current_minibatch = True
        return self._current_minibatch
    
    def set_next_minibatch(self) -> None:
        if not self._has_used_current_minibatch:
            warnings.warn("Setting the next batch in the dataset iterator, even though the "
                    "previous batch has not been used yet. Make sure you are not updating the "
                    "same dataset iterator from several endpoints (e.g. via diag datasets).")
        self._current_minibatch = next(self.iterator)
        self._has_used_current_minibatch = False


def diag_dataset_params(ds_params_list: list[DatasetParams]) -> DatasetParams:
    if not all(ds_params_list[0].fit(ds_params) for ds_params in ds_params_list):
        raise ValueError("Some domains have not matching values in `dataset_params`. It is not " +
            "possible to diag them without assigning a new set of parameters.")
    if all(ds_params.map_fn is None for ds_params in ds_params_list):
        diag_map_fn = None
    else:
        map_fns = []
        for ds_params in ds_params_list:
            map_fn = ds_params.map_fn if ds_params.map_fn is not None else lambda x: x
            map_fns.append(map_fn)
        diag_map_fn = lambda *args: [map_fns[i](args[i]) for i in range(len(map_fns))]
    diag_ds_params = copy.copy(ds_params_list[0])
    diag_ds_params.map_fn = diag_map_fn
    return diag_ds_params


def diag_domains(domains: list[Domain], dataset_params: DatasetParams = None) -> None:
    """In place."""
    for dom in domains:
        if dom.is_diagged:
            raise ValueError(f"Domain {dom.label} is already diagged. Undiag it first.")
    if dataset_params is None:
        dataset_params = diag_dataset_params([dom.dataset_params for dom in domains])
    diag_label = "diag_"+"_".join([dom.label for dom in domains])
    for dom in domains:
        dom._diag_label = diag_label
    raw_diag_dataset = tf.data.Dataset.zip(tuple(dom._raw_dataset for dom in domains))
    diag_dataset = dataset_params.apply_to_dataset(raw_diag_dataset)
    diag_dataset_iterator = DatasetIterator(diag_dataset)
    for (i, dom) in enumerate(domains):
        dom._set_diag_dataset(diag_dataset, index_in_diag_dataset=i, iterator=diag_dataset_iterator)


def undiag_domains(domains: list[Domain]) -> None:
    """In place"""
    for dom in domains:
        dom.undiag()


