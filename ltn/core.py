from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

VarLabel = str
FloatTensorLike = (
    tf.types.experimental.TensorLike
)  # to update when tf supports better type annotations


class Expression:
    def __init__(self, tensor: tf.Tensor, free_vars: list[VarLabel]) -> None:
        self.tensor: tf.Tensor = tensor
        self.free_vars: list[VarLabel] = free_vars

    def __repr__(self) -> str:
        return f"ltn.{self.__class__.__name__}(tensor={self.tensor}, free_vars={self.free_vars})"

    def _copy(self) -> Expression:
        """Copy the expression but point to the same tensor, for gradient tracking."""
        return Expression(self.tensor, self.free_vars.copy())

    def _get_axis_of_free_var(self, free_var: VarLabel) -> int:
        if free_var not in self.free_vars:
            raise ValueError(f"{free_var} is not a free variable occurring in the expression.")
        return self.free_vars.index(free_var)

    def _get_dim_of_free_var(self, free_var: VarLabel) -> tf.Tensor:
        return tf.shape(self.tensor)[self._get_axis_of_free_var(free_var)]

    def take(self, free_var: VarLabel, indices: int | list[int]) -> Expression:
        """Take elements along the axis that corresponds to `free_var`."""
        # static rank, so that the branching also works when tracing a tf.function
        rank = tf.convert_to_tensor(indices).shape.rank
        if rank == 0:
            remaining_free_vars = [v for v in self.free_vars if v != free_var]
        elif rank == 1:
            remaining_free_vars = self.free_vars.copy()
        else:
            raise ValueError("Give a single indice or a list of indices.")
        result = self._copy()
        result.tensor = tf.gather(self.tensor, indices, axis=self._get_axis_of_free_var(free_var))
        result.free_vars = remaining_free_vars
        return result


class Term(Expression):
    def __init__(self, tensor: tf.Tensor, free_vars: list[VarLabel]) -> None:
        super().__init__(tensor, free_vars=free_vars)

    def _copy(self) -> Term:
        return Term(self.tensor, self.free_vars.copy())


class Formula(Expression):
    def __init__(self, tensor: tf.Tensor, free_vars: list[VarLabel]) -> None:
        super().__init__(tensor, free_vars=free_vars)

    def _copy(self) -> Formula:
        return Formula(self.tensor, self.free_vars.copy())


class Variable(Term):
    def __init__(self, label: VarLabel, values: FloatTensorLike) -> None:
        for reserved in ["diag", "_flat"]:
            if label.startswith(reserved):
                raise ValueError(f"Labels starting with {reserved} are reserved.")
        try:
            tensor = tf.constant(values, dtype=tf.float32)
        except TypeError:
            tensor = tf.convert_to_tensor(tf.cast(values, tf.float32), dtype=tf.float32)
        if len(tensor.shape) == 0:
            raise ValueError(
                "LTN Variables must be list of values. The given values are not iterable."
            )
        if len(tensor.shape) == 1:  # ensure feature dims
            tensor = tensor[:, tf.newaxis]
        free_vars = [label]
        super().__init__(tensor, free_vars=free_vars)
        self.label: VarLabel = label
        self.locked_diag_label: str = None

    def __repr__(self) -> str:
        return (
            f"ltn.{self.__class__.__name__}(label={self.label}, tensor={self.tensor}, "
            f"free_vars={self.free_vars})"
        )

    @classmethod
    def from_constants(
        cls: Variable,
        label: VarLabel,
        constants: list[Constant],
        tape: tf.GradientTape | None = None,
    ) -> Variable:
        if not tape:
            warnings.warn(
                f"No instance of {tf.GradientTape} passed in argument when creating a LTN variable "
                "from constants. LTN cannot verify that a tape is recording. If you created the "
                "variable within the scope of a tape, or that you don't need to track weights "
                "(e.g. non-trainable constants), you can ignore this warning.",
                stacklevel=2,
            )
        else:
            if not tape._recording:
                raise ValueError("The tape is not recording.")
        dump_values = [0.0]
        variable = cls(label, dump_values)
        variable.tensor = tf.stack(as_tensors(constants))
        return variable


class Constant(Term):
    def __init__(self, value: FloatTensorLike, trainable: bool) -> None:
        self._trainable = trainable
        try:
            tensor = tf.constant(value, dtype=tf.float32)
        except TypeError:
            tensor = tf.convert_to_tensor(tf.cast(value, tf.float32), dtype=tf.float32)
        if len(tensor.shape) == 0:  # ensure feature dims
            tensor = tensor[tf.newaxis]
        if self._trainable:
            # create the variable last: indexing a variable returns a tensor, which cannot be
            # trained
            tensor = tf.Variable(tensor, trainable=True)
        free_vars = []
        super().__init__(tensor, free_vars=free_vars)

    def __repr__(self) -> str:
        return (
            f"ltn.{self.__class__.__name__}(tensor={self.tensor}, trainable={self._trainable}, "
            f"free_vars={self.free_vars})"
        )


class Proposition(Formula):
    def __init__(self, truth_value: float, trainable: bool) -> None:
        try:
            in_range = 0 <= float(truth_value) <= 1
        except (TypeError, ValueError):
            in_range = False
        if not in_range:
            raise ValueError("The truth value of a proposition should be a float in [0,1].")
        self._trainable = trainable
        if self._trainable:
            tensor = tf.Variable(
                truth_value,
                trainable=True,
                constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0),
                dtype=tf.float32,
            )
        else:
            tensor = tf.constant(truth_value, dtype=tf.float32)
        free_vars = []
        super().__init__(tensor, free_vars=free_vars)

    def __repr__(self) -> str:
        return (
            f"ltn.{self.__class__.__name__}(tensor={self.tensor}, trainable={self._trainable}, "
            f"free_vars={self.free_vars})"
        )


def _flatten_free_dims(exprs: list[Expression], in_place: bool = False) -> list[Expression]:
    if not in_place:
        exprs = [expr._copy() for expr in exprs]
    for expr in exprs:
        non_var_shape = expr.tensor.shape[len(expr.free_vars) :]
        expr.tensor = tf.reshape(expr.tensor, shape=tf.concat([[-1], non_var_shape], axis=0))
        expr.free_vars = ["_flat_" + "_".join(expr.free_vars)]
    return exprs


def _as_shape(shape: int | list[int]) -> tuple:
    """Keras 3 only accepts tuples as input shapes; also accept a single int."""
    return tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)


class _Model:
    def __init__(self, model: tf.keras.Model, with_feature_dims: bool) -> None:
        self.model: tf.keras.Model = model
        self.with_feature_dims: bool = with_feature_dims

    def __call__(self, inputs: Term | list[Term], *args: Any, **kwargs: Any) -> Expression:
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            flat_inputs = _flatten_free_dims(inputs)
            t_outputs = self.model(flat_inputs[0].tensor, *args, **kwargs)
        else:
            inputs = broadcast_exprs(inputs)
            flat_inputs = _flatten_free_dims(inputs)
            t_outputs = self.model(as_tensors(flat_inputs), *args, **kwargs)
        free_dims = tf.cast(tf.shape(inputs[0].tensor)[: len(inputs[0].free_vars)], tf.int32)
        if self.with_feature_dims:
            t_outputs = tf.reshape(
                t_outputs, tf.concat([free_dims, tf.shape(t_outputs)[1:]], axis=0)
            )
        else:
            t_outputs = tf.reshape(t_outputs, free_dims)
        t_outputs = tf.cast(t_outputs, tf.float32)
        outputs = Expression(t_outputs, inputs[0].free_vars)
        return outputs

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        if not self.model.trainable_variables:
            warnings.warn(
                "The 'trainable_variables' attribute returned an empty list. Make sure that "
                f"the weights of the layers in the {tf.keras.Model} instance have been "
                "initialized, for example by calling the model a first time.",
                stacklevel=2,
            )
        return self.model.trainable_variables


class Predicate(_Model):
    def __init__(self, model: tf.keras.Model) -> None:
        super().__init__(model, with_feature_dims=False)

    def __call__(self, inputs: Term | list[Term], *args: Any, **kwargs: Any) -> Formula:
        if not isinstance(inputs, (list, tuple)):
            if not isinstance(inputs, Term):
                raise TypeError(
                    f"The input to a LTN Predicate should be instances of {Term}. "
                    f"Got an instance of {type(inputs)} instead."
                )
        else:
            for x in inputs:
                if not isinstance(x, Term):
                    raise TypeError(
                        f"The input to a LTN Predicate should be instances of {Term}. "
                        f"Got an instance of {type(x)} instead."
                    )
        expr = super().__call__(inputs, *args, **kwargs)
        wff = Formula(expr.tensor, expr.free_vars)
        return wff

    @classmethod
    def FromLogits(
        cls,
        logits_model: tf.keras.Model,
        activation_function: str,
        with_class_indexing=True,
        **kwargs: Any,
    ) -> Predicate:
        r"""Constructor from a model that outputs logits.

        A model for a predicate `P` classifying individuals `x` into n classes
        `class_1`, ..., `class_n` will likely output n logits using the same
        hidden layers.

        This constructor allows to easily define a predicate that can be called
        using the syntax `P([x,class_i])`.

        Given `logits_model`, the model that outputs logits for `x`, the predicate will:
        1. Transform the logits into values in [0,1] using either a sigmoid of softmax activation,
        2. Return the value at the index `class_i` (if `with_class_indexing` is True).

        An important requirement of `logits_model` is that it must receive
        a list of inputs in argument. That is, `logits_model([x])` must work,
        instead of `logits_model(x)`. This is how the predicate will call it,
        even if there is a single argument.

        Args:
            logits_model (tf.keras.Model): Model that outputs logits.
                Is called using a list of inputs (`logits_model([x1,x2,...])`), even if
                there is a single input (`logits_model([x])`).
            activation_function (str): "sigmoid" or "softmax". For softmax activations,
                `with_class_indexing` is considered True.
            with_class_indexing (bool, optional): In the case of a sigmoid activation,
                whether the predicate should support the syntax `P([x,class_i])` (when True),
                or the syntax `P([x])` (when False). Defaults to True.

        Raises:
            ValueError: The activation function is not "sigmoid" or "softmax".

        Returns:
            Predicate: A `Predicate` object that can be called with the syntax `P([x,class_i])`.
        """
        if activation_function == "sigmoid":
            predicate = cls(
                _SigmoidTfModel(logits_model, with_class_indexing=with_class_indexing, **kwargs)
            )
        elif activation_function == "softmax":
            predicate = cls(_SoftmaxTfModel(logits_model, **kwargs))
        else:
            raise ValueError(
                'Computation from logits is implemented only for "sigmoid" or "softmax".'
            )
        predicate.logits_model = logits_model
        return predicate

    @classmethod
    def MLP(cls: Predicate, input_shapes, hidden_layer_sizes=(16, 16)) -> Predicate:
        inputs = [tf.keras.Input(_as_shape(shape)) for shape in input_shapes]
        flat_inputs = [layers.Flatten()(x) for x in inputs]
        hidden = layers.Concatenate()(flat_inputs) if len(flat_inputs) > 1 else flat_inputs[0]
        for units in hidden_layer_sizes:
            hidden = layers.Dense(units, activation=tf.nn.elu)(hidden)
        outputs = layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return cls(model)

    @classmethod
    def Lambda(cls: Predicate, lambda_operator: Callable) -> Predicate:
        model = tf_LambdaModel(lambda_operator)
        return cls(model)


class Function(_Model):
    def __init__(self, model: tf.keras.Model) -> None:
        super().__init__(model, with_feature_dims=True)

    def __call__(self, inputs: Term | list[Term], *args: Any, **kwargs: Any) -> Term:
        if not isinstance(inputs, (list, tuple)):
            if not isinstance(inputs, Term):
                raise TypeError(
                    f"The input to a LTN Function should be instances of {Term}. "
                    f"Got an instance of {type(inputs)} instead."
                )
        else:
            for x in inputs:
                if not isinstance(x, Term):
                    raise TypeError(
                        f"The input to a LTN Function should be instances of {Term}. "
                        f"Got an instance of {type(x)} instead."
                    )
        expr = super().__call__(inputs, *args, **kwargs)
        term = Term(expr.tensor, expr.free_vars)
        return term

    @classmethod
    def MLP(cls: Function, input_shapes, output_shape, hidden_layer_sizes=(16, 16)) -> Function:
        inputs = [tf.keras.Input(_as_shape(shape)) for shape in input_shapes]
        flat_inputs = [layers.Flatten()(x) for x in inputs]
        hidden = layers.Concatenate()(flat_inputs) if len(flat_inputs) > 1 else flat_inputs[0]
        for units in hidden_layer_sizes:
            hidden = layers.Dense(units, activation=tf.nn.elu)(hidden)
        output_shape = _as_shape(output_shape)
        output_nodes = int(np.prod(output_shape))
        flat_outputs = layers.Dense(output_nodes)(hidden)
        outputs = layers.Reshape(output_shape)(flat_outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return cls(model)

    @classmethod
    def Lambda(cls: Function, lambda_operator: Callable) -> Function:
        model = tf_LambdaModel(lambda_operator)
        return cls(model)


class tf_LambdaModel(tf.keras.Model):
    """Simple `tf.keras.Model` that implements a lambda layer."""

    def __init__(self, lambda_operator: Callable) -> None:
        super().__init__()
        self.lambda_layer = layers.Lambda(lambda_operator)

    def call(self, inputs: tf.Tensor | list[tf.Tensor]) -> tf.Tensor:
        return self.lambda_layer(inputs)


def diag(*variables: Variable) -> list[Variable]:
    for var in variables:
        if var.free_vars[0].startswith("diag_"):
            raise ValueError(
                "Trying to diag a variable that is already temporarily " + f"diagged: {var.label}."
            )
    diag_label = "diag_" + "_".join([var.label for var in variables])
    for var in variables:
        var.free_vars = [diag_label]
    return variables


def undiag(*variables: Variable) -> list[Variable]:
    for var in variables:
        var.free_vars = [var.locked_diag_label] if var.locked_diag_label else [var.label]
    return variables


def diag_lock(*variables: Variable) -> list[Variable]:
    """In place"""
    for var in variables:
        if var.free_vars[0].startswith("diag"):
            raise ValueError(
                "Trying to diaglock a variable that is temporarily diagged: "
                + f"{var.label}.\nCall `diag_lock` on variables when they are undiagged."
            )
    diag_label = "diaglock_" + "_".join([var.label for var in variables])
    for var in variables:
        var.locked_diag_label = diag_label
        var.free_vars = [var.locked_diag_label]


def as_tensors(expressions: list[Expression]) -> list[tf.Tensor]:
    return [expr.tensor for expr in expressions]


def broadcast_exprs(exprs: list[Expression], in_place: bool = False) -> list[Expression]:
    # measure dimensions for each free variable
    free_var_to_dim = {}
    for expr in exprs:
        for free_var in expr.free_vars:
            free_var_to_dim[free_var] = expr._get_dim_of_free_var(free_var)
    free_vars = list(free_var_to_dim.keys())
    # broadcast
    if not in_place:
        exprs = [expr._copy() for expr in exprs]
    for expr in exprs:
        free_vars_in_arg = list(expr.free_vars)
        free_vars_not_in_arg = list(set(free_vars).difference(free_vars_in_arg))
        for new_free_var in free_vars_not_in_arg:
            new_idx = len(free_vars_in_arg)
            expr.tensor = tf.expand_dims(expr.tensor, axis=new_idx)
            expr.tensor = tf.repeat(expr.tensor, free_var_to_dim[new_free_var], axis=new_idx)
            free_vars_in_arg.append(new_free_var)
        perm = [free_vars_in_arg.index(free_var) for free_var in free_vars] + list(
            range(len(free_vars_in_arg), len(expr.tensor.shape))
        )
        expr.tensor = tf.transpose(expr.tensor, perm=perm)
        expr.free_vars = free_vars
    return exprs


class Wrapper_Connective:
    def __init__(self, connective_op: Callable) -> None:
        self.connective_op = connective_op

    def __call__(self, *wffs: Formula, **kwargs: Any) -> Formula:
        for x in wffs:
            if not isinstance(x, Formula):
                raise TypeError(
                    f"The operands of a LTN connective should be instances of {Formula}. "
                    f"Got an instance of {type(x)} instead."
                )
        wffs = broadcast_exprs(wffs)
        try:
            t_result = self.connective_op(*as_tensors(wffs), **kwargs)
        except tf.errors.InvalidArgumentError as err:
            shapes = ", ".join(str(wff.tensor.shape) for wff in wffs)
            free_vars = ", ".join(str(wff.free_vars) for wff in wffs)
            raise ValueError(
                f"Could not connect formulas with shapes [{shapes}] "
                f"and free variables [{free_vars}]."
            ) from err
        result = Formula(t_result, wffs[0].free_vars)
        return result


class Wrapper_Quantifier:
    def __init__(self, aggreg_op: Callable, semantics: str) -> None:
        self.aggreg_op = aggreg_op
        if semantics not in ["forall", "exists"]:
            raise ValueError('The semantics for the quantifier should be "forall" or "exists".')
        self.semantics = semantics

    def __call__(
        self,
        variables: list[Variable] | Variable,
        wff: Formula,
        mask: Formula | None = None,
        **kwargs: Any,
    ) -> Formula:
        variables = [variables] if not isinstance(variables, (list, tuple)) else variables
        for x in variables:
            if not isinstance(x, Variable):
                raise TypeError(
                    f"The quantified variables should be instances of {Variable}. "
                    f"Got an instance of {type(x)} instead."
                )
        if not isinstance(wff, Formula):
            raise TypeError(
                f"The quantified expression should be an instance of {Formula}. "
                f"Got an instance of {type(wff)} instead."
            )
        aggreg_vars = set([var.free_vars[0] for var in variables])
        if mask is not None:
            if not isinstance(mask, Formula):
                raise TypeError(
                    f"The mask argument should be an instance of {Formula}. "
                    f"Got an instance of {type(mask)} instead."
                )
            mask = transpose_free_vars(
                mask,
                new_var_order=[
                    var for var in mask.free_vars if var not in aggreg_vars
                ]  # important to put aggreg dims last,
                + [var for var in mask.free_vars if var in aggreg_vars],
            )  # to keep other dims in the ragged result
            wff = broadcast_wff_and_mask(wff, mask)
            mask.tensor = tf.cast(mask.tensor, tf.bool)
            aggreg_axes = [wff.free_vars.index(var) for var in aggreg_vars]
            aggreg_axes_in_mask = [
                mask.free_vars.index(var) for var in aggreg_vars if var in mask.free_vars
            ]
            if aggreg_axes_in_mask:
                t_ragged_wff = tf.ragged.boolean_mask(wff.tensor, mask.tensor)
                t_result = self.aggreg_op(t_ragged_wff, axis=aggreg_axes, **kwargs)
                if isinstance(t_result, tf.RaggedTensor):
                    t_result = t_result.to_tensor()
            else:
                # The mask only guards free vars of the result, so the quantified vars keep their
                # whole domain. (`boolean_mask` would compact the last axis of the mask, which is
                # then a free var of the result.)
                t_result = self.aggreg_op(wff.tensor, axis=aggreg_axes, **kwargs)

            non_empty_vars = (
                tf.reduce_sum(tf.cast(mask.tensor, tf.int32), axis=aggreg_axes_in_mask) != 0
            )
            # the result has the free vars of the mask first, then those of the wff only;
            # add trailing axes so that non_empty_vars broadcasts over the latter
            n_vars_not_in_mask = len(
                [
                    var
                    for var in wff.free_vars
                    if var not in aggreg_vars and var not in mask.free_vars
                ]
            )
            for _ in range(n_vars_not_in_mask):
                non_empty_vars = tf.expand_dims(non_empty_vars, axis=-1)
            empty_semantics = 1.0 if self.semantics == "forall" else 0

            t_result = tf.where(non_empty_vars, t_result, empty_semantics)
        else:
            aggreg_axes = [wff.free_vars.index(var) for var in aggreg_vars]
            empty_domain = tf.reduce_any(tf.gather(tf.shape(wff.tensor), aggreg_axes) == 0)
            empty_semantics = 1.0 if self.semantics == "forall" else 0.0
            # Do not evaluate the aggregator on an empty domain: even replacing
            # its NaN result afterwards can leave NaN gradients. The empty sum
            # preserves the result shape and a zero gradient to the input.
            t_result = tf.cond(
                empty_domain,
                lambda: (
                    tf.reduce_sum(
                        wff.tensor, axis=aggreg_axes, keepdims=kwargs.get("keepdims", False)
                    )
                    + tf.cast(empty_semantics, wff.tensor.dtype)
                ),
                lambda: self.aggreg_op(wff.tensor, axis=aggreg_axes, **kwargs),
            )
        free_vars_remaining = [var for var in wff.free_vars if var not in aggreg_vars]
        result = Formula(t_result, free_vars_remaining)
        undiag(*variables)
        return result


class Wrapper_Formula_Aggregator:
    def __init__(self, aggreg_op: Callable) -> None:
        self.aggreg_op = aggreg_op

    def __call__(self, wffs: list[Formula], **kwargs: Any) -> Formula:
        for wff in wffs:
            if wff.free_vars:  # list not empty
                raise ValueError("Some formulas still contain free variables.")
        t_result = self.aggreg_op(tf.stack(as_tensors(wffs)), **kwargs)
        result = Formula(t_result, free_vars=[])
        return result


def broadcast_wff_and_mask(wff: Formula, mask: Formula) -> Formula:
    """Broadcast the wff to include all vars in mask; put the vars of the mask in the first axes"""
    wff = wff._copy()
    # 1. broadcast wff with vars that are in the mask but not yet in the formula
    mask_vars_not_in_wff = [var for var in mask.free_vars if var not in wff.free_vars]
    for var in mask_vars_not_in_wff:
        new_idx = len(wff.free_vars)
        wff.tensor = tf.expand_dims(wff.tensor, axis=new_idx)
        wff.tensor = tf.repeat(wff.tensor, mask._get_dim_of_free_var(var), axis=new_idx)
        wff.free_vars.append(var)
    # 2. transpose wff so that the masked vars on the first axes
    vars_not_in_mask = [var for var in wff.free_vars if var not in mask.free_vars]
    wff = transpose_free_vars(wff, new_var_order=mask.free_vars + vars_not_in_mask)
    return wff


def transpose_free_vars(
    expr: Expression, new_var_order: list[VarLabel], in_place: bool = False
) -> Expression:
    perm = [expr.free_vars.index(var) for var in new_var_order]
    perm.extend(range(len(expr.free_vars), len(expr.tensor.shape)))
    if not in_place:
        expr = expr._copy()
    expr.tensor = tf.transpose(expr.tensor, perm)
    expr.free_vars = new_var_order
    return expr


class _SigmoidTfModel(tf.keras.Model):
    def __init__(
        self, logits_model: tf.keras.Model, with_class_indexing: bool = False, **kwargs: Any
    ) -> None:
        """with_class_indexing: If true, must have last axis (-1) for indexing classes.
        logits_model : must accept list of inputs
        """
        # the name Keras 3 derives from the private class name, "__sigmoid_tf_model",
        # is not a valid name scope inside a `tf.function`
        super().__init__(name="sigmoid_tf_model")
        self.logits_model = logits_model
        self.call = (
            self._call_with_class_indexing
            if with_class_indexing
            else self._call_without_class_indexing
        )

    def _call_without_class_indexing(
        self, inputs: list[tf.Tensor], *args: Any, **kwargs: Any
    ) -> tf.Tensor:
        logit = self.logits_model(inputs, *args, **kwargs)
        truth_degree = tf.math.sigmoid(logit)
        return truth_degree

    def _call_with_class_indexing(
        self, inputs: list[tf.Tensor], *args: Any, **kwargs: Any
    ) -> tf.Tensor:
        """inputs[-1] are the classes to index"""
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs, *args, **kwargs)
        truth_degrees = tf.math.sigmoid(logits)
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(truth_degrees, indices, batch_dims=1)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return self.logits_model.trainable_variables


class _SoftmaxTfModel(tf.keras.Model):
    def __init__(self, logits_model: tf.keras.Model, **kwargs: Any) -> None:
        """logits_model: Must have last axis for classes, even if only one class."""
        super().__init__(name="softmax_tf_model")  # see _SigmoidTfModel
        self.logits_model = logits_model

    def call(self, inputs: list[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        """inputs[-1] are the classes to index"""
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs, *args, **kwargs)
        truth_degrees = tf.nn.softmax(logits)
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(truth_degrees, indices, batch_dims=1)

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        return self.logits_model.trainable_variables
