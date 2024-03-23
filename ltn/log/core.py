from __future__ import annotations
from typing import Optional, Union, List, Callable, Any

import tensorflow as tf

from ltn import core

class LogFormula(core.Expression):
    def __init__(self, tensor: tf.Tensor, free_vars: List[core.VarLabel]) -> None:
        super().__init__(tensor, free_vars=free_vars)
    
    def _copy(self) -> LogFormula:
        return LogFormula(self.tensor, self.free_vars.copy())

    def sat(self) -> core.Formula:
        sat_tensor = tf.math.exp(self.tensor)
        return core.Formula(sat_tensor, self.free_vars.copy())

class Predicate(core.Predicate):
    def __init__(self, model: tf.keras.Model) -> None:
        super().__init__(model)
        self.log_ltnmodel = None
        self.nlog_ltnmodel = None

    @classmethod
    def FromLogits(cls, logits_model: tf.keras.Model, activation_function: str, 
            with_class_indexing=False, **kwargs: Any) -> Predicate:
        """ with_class_indexing: If true, must have last axis (-1) for indexing classes.
        """
        if activation_function == "sigmoid":
            predicate = cls(_SigmoidTfModel(logits_model, with_class_indexing=with_class_indexing, **kwargs))
            predicate.log_ltnmodel = core._Model(_LogSigmoidTfModel(logits_model, with_class_indexing=with_class_indexing, **kwargs), with_feature_dims=False)
            predicate.nlog_ltnmodel = core._Model(_NlogSigmoidTfModel(logits_model, with_class_indexing=with_class_indexing, **kwargs), with_feature_dims=False)
        elif activation_function == "softmax":
            predicate = cls(_SoftmaxTfModel(logits_model, **kwargs))
            predicate.log_ltnmodel = core._Model(_LogSoftmaxTfModel(logits_model, **kwargs), with_feature_dims=False)
            predicate.nlog_ltnmodel = core._Model(_NlogSoftmaxTfModel(logits_model, **kwargs), with_feature_dims=False)
        else:
            raise ValueError("Computation from logits is implemented only for \"sigmoid\" or \"softmax\".")
        predicate.logits_model = logits_model
        return predicate

    def log(self, inputs: Union[core.Term, List[core.Term]], *args: Any, **kwargs: Any) -> LogFormula:
        if self.log_ltnmodel is not None: # if there is a special implementation for the positive log model, use it
            log_expr = self.log_ltnmodel(inputs, *args, **kwargs)
        else:
            log_expr = self(inputs, *args, **kwargs)
            log_expr.tensor = tf.clip_by_value(log_expr.tensor, 0.01, 1.0) # cannot do log(0)
            log_expr.tensor = tf.math.log(log_expr.tensor)
        log_formula = LogFormula(log_expr.tensor, log_expr.free_vars.copy())
        return log_formula

    def nlog(self, inputs: Union[core.Term, List[core.Term]], *args: Any, **kwargs: Any) -> LogFormula:
        if self.nlog_ltnmodel is not None: # if there is a special implementation for the negative log model, use it
            log_expr = self.nlog_ltnmodel(inputs, *args, **kwargs)
        else:
            log_expr = self(inputs, *args, **kwargs)
            log_expr.tensor = tf.clip_by_value(log_expr.tensor, 0., 0.99) # cannot do log(0)
            log_expr.tensor = tf.math.log(1 - log_expr.tensor)
        log_formula = LogFormula(log_expr.tensor, log_expr.free_vars.copy())
        return log_formula

class _SigmoidTfModel(tf.keras.Model):
    def __init__(self, logits_model: tf.keras.Model, with_class_indexing: bool = False, **kwargs: Any) -> None:
        """ with_class_indexing: If true, must have last axis (-1) for indexing classes.
        logits_model : must accept list of inputs
         """
        super().__init__()
        self.logits_model = logits_model
        self.call = self._call_with_class_indexing if with_class_indexing else self._call_without_class_indexing

    def _call_without_class_indexing(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        logit = self.logits_model(inputs)
        truth_degree = tf.math.sigmoid(logit)    
        return truth_degree

    def _call_with_class_indexing(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        """ inputs[-1] are the classes to index """
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs)
        truth_degrees = tf.math.sigmoid(logits)
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(truth_degrees, indices, batch_dims=1)

class _LogSigmoidTfModel(tf.keras.Model):
    def __init__(self, logits_model: tf.keras.Model, with_class_indexing: bool, **kwargs: Any) -> None:
        """
        logits_model : must accept list of inputs
        with_class_indexing: If true, the inputs must have a last axis (-1) value for indexing classes.
        """
        super().__init__()
        self.logits_model = logits_model
        self.call = self._call_with_class_indexing if with_class_indexing else self._call_without_class_indexing

    def _call_without_class_indexing(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        logit = self.logits_model(inputs)
        log_truth_degree = tf.math.log_sigmoid(logit)    
        return log_truth_degree

    def _call_with_class_indexing(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        """ inputs[-1] are the classes to index """
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs)
        log_truth_degrees = tf.math.log_sigmoid(logits)
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(log_truth_degrees, indices, batch_dims=1)

class _NlogSigmoidTfModel(tf.keras.Model):
    def __init__(self, logits_model: tf.keras.Model, with_class_indexing: bool, **kwargs: Any) -> None:
        """
        logits_model: Must have last axis for classes, even if only one class. 
        with_class_indexing: If true, the inputs must have a last axis (-1) value for indexing classes.
        """
        super().__init__()
        self.logits_model = logits_model
        self.call = self._call_with_class_indexing if with_class_indexing else self._call_without_class_indexing
        
    def _call_without_class_indexing(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        logit = self.logits_model(inputs)
        log_truth_degree = tf.math.log_sigmoid(logit) - logit
        return log_truth_degree

    def _call_with_class_indexing(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        """ inputs[-1] are the classes to index """
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs)
        log_truth_degrees = tf.math.log_sigmoid(logits) - logits
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(log_truth_degrees, indices, batch_dims=1)

class _SoftmaxTfModel(tf.keras.Model):
    def __init__(self, logits_model: tf.keras.Model, **kwargs: Any) -> None:
        """ logits_model: Must have last axis for classes, even if only one class. """
        super().__init__()
        self.logits_model = logits_model
        
    def call(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        """ inputs[-1] are the classes to index """
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs)
        truth_degrees = tf.nn.softmax(logits)
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(truth_degrees, indices, batch_dims=1)

class _LogSoftmaxTfModel(tf.keras.Model):
    def __init__(self, logits_model: tf.keras.Model, **kwargs: Any) -> None:
        """ logits_model: Must have last axis for classes, even if only one class. """
        super().__init__()
        self.logits_model = logits_model
        
    def call(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        """ inputs[-1] are the classes to index """
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs)
        log_truth_degrees = tf.nn.log_softmax(logits)
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(log_truth_degrees, indices, batch_dims=1)

class _NlogSoftmaxTfModel(tf.keras.Model):
    def __init__(self, logits_model: tf.keras.Model, **kwargs: Any) -> None:
        """ logits_model: Must have last axis for classes, even if only one class. """
        super().__init__()
        self.logits_model = logits_model
        
    def call(self, inputs: List[tf.Tensor], *args: Any, **kwargs: Any) -> tf.Tensor:
        """ inputs[-1] are the classes to index """
        logits_model_inputs, indices = inputs[:-1], inputs[-1]
        logits = self.logits_model(logits_model_inputs) # None x i
        # LogSumExp of other logits:
        # [1      [[1 2 3]     [[  2 3]     [LSE(2,3)
        #  2   ->  [1 2 3]  ->  [1   3]  ->  LSE(1,3)
        #  3]      [1 2 3]]     [1 2  ]]     LSE(1,2)]
        #
        x = tf.expand_dims(logits,axis=-2)
        x = tf.repeat(x,axis=-2,repeats=tf.shape(x)[-1])
        not_diagonal = tf.math.logical_not(tf.cast(tf.linalg.diag(logits),tf.bool))
        x = tf.ragged.boolean_mask(x, not_diagonal).to_tensor() # TODO: simpler implementation
        logsumexp_other_logits = tf.reduce_logsumexp(x, axis=-1, keepdims=False)
        log_truth_degrees = tf.nn.log_softmax(logits) + logsumexp_other_logits - logits
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(log_truth_degrees, indices, batch_dims=1)

class Wrapper_Connective:
    def __init__(self, connective_op: Callable) -> None:
        self.connective_op = connective_op

    def __call__(self, *wffs: LogFormula, **kwargs: Any) -> LogFormula:
        for x in wffs:
            if not isinstance(x, LogFormula):
                raise TypeError("The operands of a log-LTN connective should be instances of %s. \
                        Got an instance of %s instead." % (LogFormula, type(x)))
        wffs = core.broadcast_exprs(wffs)
        try:
            t_result = self.connective_op(*core.as_tensors(wffs), **kwargs)
        except tf.errors.InvalidArgumentError:
            raise ValueError("Could not connect formulas with shapes [%s] and free variables [%s]."
                % (', '.join(map(str,[wff.tensor.shape for wff in wffs])),
                ', '.join(map(str,[wff.free_vars for wff in wffs])))
            )
        result = LogFormula(t_result, wffs[0].free_vars)
        return result


class Wrapper_Quantifier:
    def __init__(self, aggreg_op: Callable, semantics: str) -> None:
        self.aggreg_op = aggreg_op
        if semantics not in ["forall","exists"]:
            raise ValueError("The semantics for the quantifier should be \"forall\" or \"exists\".")
        self.semantics = semantics

    def __call__(self, 
            variables: Union[List[core.Variable],core.Variable],
            wff: LogFormula,
            mask: Optional[core.Formula] = None,
            **kwargs: Any) -> LogFormula:
        variables = [variables] if not isinstance(variables,(list,tuple)) else variables
        for x in variables:
            if not isinstance(x, core.Variable):
                raise TypeError("The quantified variables should be instances of %s. "\
                        "Got an instance of %s instead." % (core.Variable, type(x)))
        if not isinstance(wff, LogFormula):
            raise TypeError("The quantified expression in log-LTN should be an instance of %s. "\
                    "Got an instance of %s instead." % (LogFormula, type(x)))
        aggreg_vars = set([var.free_vars[0] for var in variables])
        if mask is not None:
            if not isinstance(mask, core.Formula):
                raise TypeError("The mask argument should be an instance of %s. "\
                        "Got an instance of %s instead." % (core.Formula, type(mask)))
            mask = core.transpose_free_vars(mask, 
                    new_var_order = [var for var in mask.free_vars if var not in aggreg_vars]   # important to put aggreg dims last,
                            + [var for var in mask.free_vars if var in aggreg_vars])            # to keep other dims in the ragged result 
            wff = core.broadcast_wff_and_mask(wff, mask)
            mask.tensor = tf.cast(mask.tensor, tf.bool)
            t_ragged_wff = tf.ragged.boolean_mask(wff.tensor, mask.tensor)
            aggreg_axes = [wff.free_vars.index(var) for var in aggreg_vars]
            t_result = self.aggreg_op(t_ragged_wff, axis=aggreg_axes, **kwargs)
            if isinstance(t_result, tf.RaggedTensor):
                t_result = t_result.to_tensor()
            
            aggreg_axes_in_mask = [mask.free_vars.index(var) for var in aggreg_vars 
                    if var in mask.free_vars]
            non_empty_vars = tf.reduce_sum(tf.cast(mask.tensor,tf.int32), axis=aggreg_axes_in_mask) != 0
            empty_semantics = 0. if self.semantics == "forall" else -1e5
            
            t_result = tf.where(
                non_empty_vars,
                t_result,
                empty_semantics
            )
        else:
            aggreg_axes = [wff.free_vars.index(var) for var in aggreg_vars]
            t_result = self.aggreg_op(wff.tensor, axis=aggreg_axes, **kwargs)
        free_vars_remaining = [var for var in wff.free_vars if var not in aggreg_vars]
        result = LogFormula(t_result, free_vars_remaining)
        core.undiag(*variables)
        return result

class Wrapper_Formula_Aggregator:
    def __init__(self, aggreg_op: Callable) -> None:
        self.aggreg_op = aggreg_op

    def __call__(self, wffs: List[LogFormula], broadcast_formulas: bool = False, **kwargs: Any) -> LogFormula:
        if broadcast_formulas:
            wffs = core.broadcast_exprs(wffs)
        else:
            for wff in wffs:
                if wff.free_vars: # list not empty
                    raise ValueError("Error when aggregating the formulas: some formulas still contain free variables. "\
                        "If this is intended, consider broadcasting the formulas using the parameter `broadcast_formulas=True`.")
        try:
            t_result = self.aggreg_op(tf.stack(core.as_tensors(wffs)), axis=0, **kwargs)
        except tf.errors.InvalidArgumentError:
            raise ValueError("Could not connect formulas with shapes [%s] and free variables [%s]. "
                % (', '.join(map(str,[wff.shape for wff in wffs])),
                ', '.join(map(str,[wff.free_vars for wff in wffs])))
            )
        result = LogFormula(t_result, free_vars=wffs[0].free_vars)
        return result