import tensorflow as tf
from tensorflow.keras import layers
import logging

def constant(value, trainable=False):
    """Returns a Tensor with the same values and contents as feed, that can be used as a ltn constant.
    
    A ltn constant denotes an individual grounded as a tensor in the Real field. 
    The individual can be pre-defined (data point) or learnable (embedding).

    Args:
        value: A value to feed in the tensor.
        trainable: If True, `tf.GradientTapes` automatically watch the ltn constant. Defaults to False.
    """
    #if value.dtype != tf.float32:
    #    logging.getLogger(__name__).info("Casting constant to tf.float32")
    result = tf.cast(value, tf.float32)
    if trainable:
        result = tf.Variable(result, trainable=True)
    else:
        result = tf.constant(result)
    result.active_doms = []
    return result

def variable(label, feed):
    """Returns a Tensor with the same values and contents as feed, that can be used as a ltn variable. 
    
    A ltn variable denotes a sequence of individuals.
    Axis 0 is the batch dimension: if `x` is an `ltn.variable`, `x[0]` gives the first individual,
    `x[1]` gives the second individual, and so forth, the usual way.

    Args:
        label: string. In ltn, variables need to be labelled.
        feed: A value to feed in a constant tensor.
            Alternatively, a tensor to use as is (with some dynamically added attributes).
    """
    if label.startswith("diag"):
        raise ValueError("Labels starting with diag are reserved.")
    if isinstance(feed, tf.Tensor):
        result = feed
    else:
        result = tf.constant(feed)
    #if result.dtype != tf.float32 :
    #    logging.getLogger(__name__).info("Casting variable to tf.float32")
    if len(tf.shape(result)) == 1:
        result = result[:, tf.newaxis]
    result = tf.cast(result, tf.float32)
    result.latent_dom = label
    result.active_doms = [label]
    return result

class Predicate(tf.keras.Model):
    """Predicate class for ltn.

    A ltn predicate is a mathematical function (either pre-defined or learnable) that maps
    from some n-ary domain of individuals to a real from [0,1] that can be interpreted as a truth value.
    Examples of predicates can be similarity measures, classifiers, etc.
    
    Predicates can be defined using any operations in Tensorflow. They can be linear functions, Deep Neural Networks, and so forth.

    An ltn predicate implements a `tf.keras.Model` instance that can "broadcast" ltn terms as follows:
    1. Evaluating a predicate with one variable of n individuals yields n output values, 
    where the i-th output value corresponds to the term calculated with the i-th individual.
    2. Evaluating a predicate with k variables (x1,...,xk) with respectively n1,...,nK 
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor 
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable. 
    The tensor output by a predicate has a dynamically added attribute `active_doms` 
    that tells which axis corresponds to which variable (using the label of the variable).

    Attributes:
        model: The wrapped tensorflow model, without the ltn-specific broadcasting. 
    """
    def __init__(self, model):
        """Inits the ltn predicate with the given tf.keras.Model instance, 
        wrapping it with the broadcasting mechanism."""
        super(Predicate, self).__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        """Encapsulates the "self.model.__call__" to handle the broadcasting.
        
        Args:
            inputs: tensor or list of tensors that are ltn terms (ltn variable, ltn constant or 
                    output of a ltn functions). 
        Returns:
            outputs: tensor of truth values, with dimensions s.t. each variable corresponds to one axis.
        """
        if not isinstance(inputs,(list,tuple)):
            inputs, doms, dims_0 = cross_args([inputs],flatten_dim0=True)
            inputs = inputs[0]
        else:
            inputs, doms, dims_0 = cross_args(inputs, flatten_dim0=True)
        outputs = self.model(inputs, *args, **kwargs)
        dims_0 = tf.cast(dims_0,tf.int32) # is a fix when dims_0 is an empty list
        outputs = tf.reshape(outputs, dims_0)
        outputs = tf.cast(outputs,tf.float32)
        outputs.active_doms = doms
        return outputs

    @classmethod
    def Lambda(cls, lambda_operator):
        """Constructor that takes in argument a lambda function. It is appropriate for small 
        non-trainable mathematical operations that return a value in [0,1]."""
        model = LambdaModel(lambda_operator)
        return cls(model)

    @classmethod
    def MLP(cls, input_shapes, hidden_layer_sizes=(16,16)):
        inputs = [tf.keras.Input(shape) for shape in input_shapes]
        flat_inputs = [layers.Flatten()(x) for x in inputs]
        hidden = layers.Concatenate()(flat_inputs) if len(flat_inputs) > 1 else flat_inputs[0]
        for units in hidden_layer_sizes:
            hidden = layers.Dense(units,activation=tf.nn.elu)(hidden)
        outputs = layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return cls(model)

class Function(tf.keras.Model):
    """Function class for LTN.

    A ltn function is a mathematical function (pre-defined or learnable) that maps
    n individuals to one individual in the tensor domain. 
    Examples of functions can be distance functions, regressors, etc. 

    Functions can be defined using any operations in Tensorflow. 
    They can be linear functions, Deep Neural Networks, and so forth.

    An ltn function implements a `tf.keras.Model` instance that can "broadcast" ltn terms as follows:
    1. Evaluating a term with one variable of n individuals yields n output values, 
    where the i-th output value corresponds to the term calculated with the i-th individual.
    2. Evaluating a term with k variables (x1,...,xk) with respectively n1,...,nK 
    individuals each, yields a result with n1*...*nk values. The result is organized in a tensor 
    where the first k dimensions can be indexed to retrieve the outcome(s) that correspond to each variable. 
    The tensor output by a predicate has a dynamically added attribute `active_doms` 
    that tells which axis corresponds to which variable (using the label of the variable).

    Attributes:
        model: The wrapped tensorflow model, without the ltn-specific broadcasting. 
    """
    def __init__(self, model):
        """Inits the ltn function with the given tf.keras.Model instance, 
        wrapping it with the broadcasting mechanism."""
        super(Function, self).__init__()
        self.model = model
    
    def call(self, inputs, *args, **kwargs):
        """Encapsulates the "self.model.__call__" to handle the broadcasting.
        
        Args:
            inputs: tensor or list of tensors that are ltn terms (ltn variable, ltn constant or 
                    output of a ltn functions). 
        Returns:
            outputs: tensor of terms, with dimensions s.t. each variable corresponds to one axis.
        """
        if not isinstance(inputs,(list,tuple)):
            inputs, doms, dims_0 = cross_args([inputs],flatten_dim0=True)
            inputs = inputs[0]
        else:
            inputs, doms, dims_0 = cross_args(inputs, flatten_dim0=True)
        outputs = self.model(inputs, *args, **kwargs)
        dims_0 = tf.cast(dims_0,tf.int32) # is a fix when dims_0 is an empty list
        outputs = tf.reshape(outputs, tf.concat([dims_0,outputs.shape[1::]],axis=0))
        outputs = tf.cast(outputs,tf.float32)
        outputs.active_doms = doms
        return outputs

    @classmethod
    def MLP(cls, input_shapes, output_shape, hidden_layer_sizes=(16,16)):
        inputs = [tf.keras.Input(shape) for shape in input_shapes]
        flat_inputs = [layers.Flatten()(x) for x in inputs]
        hidden = layers.Concatenate()(flat_inputs) if len(flat_inputs) > 1 else flat_inputs[0]
        for units in hidden_layer_sizes:
            hidden = layers.Dense(units,activation=tf.nn.elu)(hidden)
        flat_outputs = layers.Dense(tf.math.reduce_prod(output_shape))(hidden)
        outputs = layers.Reshape(output_shape)(flat_outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return cls(model)

    @classmethod
    def Lambda(cls, lambda_operator):
        """Constructor that takes in argument a lambda function. It is appropriate for small 
        non-trainable mathematical operations."""
        model = LambdaModel(lambda_operator)
        return cls(model)

class LambdaModel(tf.keras.Model):
    """ Simple `tf.keras.Model` that implements a lambda layer.
    Used in `ltn.Predicate.Lambda` and `ltn.Function.Lambda`. 
    """
    def __init__(self, lambda_operator):
        super(LambdaModel, self).__init__()
        self.lambda_layer = layers.Lambda(lambda_operator)
    
    def call(self, inputs):
        return self.lambda_layer(inputs)

def proposition(truth_value, trainable=False):
    """Returns a rank-0 Tensor with the given truth value, whose output is constrained in [0,1],
    that can be used as a proposition in ltn formulas.

    Args:
        truth_value: A float in [0,1].
        trainable: If True, `tf.GradientTapes` automatically watch the ltn constant. Defaults to False.
    """
    try:
        assert 0 <= float(truth_value) <= 1
    except:
        raise ValueError("The truth value of a proposition should be a float in [0,1].")
    if trainable:
        result = tf.Variable(truth_value, trainable=True, 
                constraint=lambda x: tf.clip_by_value(x, 0., 1.))
    else:
        result = tf.constant(truth_value)
    if result.dtype != tf.float32:
        logging.getLogger(__name__).info("Casting proposition to tf.float32")
        result = tf.cast(result, tf.float32)
    result.active_doms = []
    return result
    
def diag(*variables):
    """Sets the given ltn variables for diagonal quantification (no broadcasting between these variables).

    Given 2 (or more) ltn variables, there are scenarios where one wants to express statements about
    specific pairs (or tuples) only, such that the i-th tuple contains the i-th instances of the variables. 
    We allow this using `ltn.diag`. 
    Note: diagonal quantification assumes that the variables have the same number of individuals.

    Given a predicate `P(x,y)` with two variables `x` and `y`,
    the usual broadcasting followed by an aggregation would compute (in Python pseudo-code):
        ```
        for i,x_i in enumerate(x):
            for j,y_j in enumerate(y):
                results[i,j]=P(x_i,y_i)
        aggregate(results)
        ```
    In contrast, diagonal quantification would compute:
        ```
        for i,(x_i, y_i) in enumerate(zip(x,y)):
            results[i].append(P(x_i,y_i))
        aggregate(results)
        ```
    Ltn computes only the "zipped" results.
    """
    diag_dom = "diag_"+"_".join([var.latent_dom for var in variables])
    for var in variables:
        var.active_doms = [diag_dom]
    return variables

def undiag(*variables):
    """Resets the usual broadcasting strategy for the given ltn variables.

    In practice, `ltn.diag` is designed to be used with quantifiers. 
    Every quantifier automatically calls `ltn.undiag` after the aggregation is performed, 
    so that the variables keep their normal behavior outside of the formula.

    Therefore, it is recommended to use `ltn.diag` only in quantified formulas as follows:
        ```
        Forall(ltn.diag(x,l), C([x,l]))
        ```
    """
    for var in variables:
        var.active_doms = [var.latent_dom]
    return variables

def get_dim0_of_dom(wff, dom):
    """Returns the number of values that the domain takes in the expression. 
    """
    return tf.shape(wff)[wff.active_doms.index(dom)]

def cross_args(args, flatten_dim0=False):
    """
    ...

    Args:
        args: list of tensor inputs to arrange. These can be ltn variables, constants,
            functions, predicates, or any expression built on those.
        flatten_dim0: if True, .
    """
    doms_to_dim0 = {}
    for arg in args:
        for dom in arg.active_doms:
            doms_to_dim0[dom] = get_dim0_of_dom(arg, dom)
    doms = list(doms_to_dim0.keys())
    dims0 = list(doms_to_dim0.values())
    crossed_args = []
    for arg in args:
        doms_in_arg = list(arg.active_doms)
        doms_not_in_arg = list(set(doms).difference(doms_in_arg))
        for new_dom in doms_not_in_arg:
            new_idx = len(doms_in_arg)
            arg = tf.expand_dims(arg, axis=new_idx)
            arg = tf.repeat(arg, doms_to_dim0[new_dom], axis=new_idx)
            doms_in_arg.append(new_dom)
        perm = [doms_in_arg.index(dom) for dom in doms] + list(range(len(doms_in_arg),len(arg.shape)))
        arg = tf.transpose(arg, perm=perm)
        arg.active_doms = doms
        if flatten_dim0:
            non_doms_shape = arg.shape[len(doms_in_arg)::]
            arg = tf.reshape(arg, shape=tf.concat([[-1],non_doms_shape],axis=0))
        crossed_args.append(arg)
    return crossed_args, doms, dims0

class Wrapper_Connective:
    """Class to wrap binary connective operators to use them within ltn formulas.
    
    LTN suppports various logical connectives. They are grounded using fuzzy semantics. 
    We have implemented some common fuzzy logic operators using tensorflow primitives in `ltn.fuzzy_ops`. 

    The wrapper ltn.Wrapper_Connective allows to use the operators with LTN formulas. 
    It takes care of combining sub-formulas that have different variables appearing in them 
    (the sub-formulas may have different dimensions that need to be "broadcasted").

    Attributes:
        _connective_op: The original binary connective operator (without broadcasting).
    """
    def __init__(self, connective_op):
        self._connective_op = connective_op

    def __call__(self, *wffs, **kwargs):
        wffs, doms, _ = cross_args(wffs)
        try:
            result = self._connective_op(*wffs, **kwargs)
        except tf.errors.InvalidArgumentError:
            raise ValueError("Could not connect arguments with shapes [%s] and respective doms [%s]."
                % (', '.join(map(str,[wff.shape for wff in wffs])),
                ', '.join(map(str,[wff.active_doms for wff in wffs])))
            )
        result.active_doms = doms
        return result

class Wrapper_Quantifier:
    """Class to wrap binary connective operators to use them within ltn formulas.

    LTN suppports universal and existential quantification. They are grounded using aggregation operators. 
    We have implemented some common aggregators using tensorflow primitives in `ltn.fuzzy_ops`.

    The wrapper allows to use the operators with LTN formulas. 
    It takes care of selecting the tensor dimensions to aggregate, given some variables in arguments.
    Additionally, boolean conditions (`mask_fn`,`mask_vars`) can be used for guarded quantification.

    Attributes:
        _aggreg_op: The original aggregation operator.
    """
    def __init__(self, aggreg_op, semantics):
        self._aggreg_op = aggreg_op
        if semantics not in ["forall","exists"]:
            raise ValueError("The semantics for the quantifier should be \"forall\" or \"exists\".")
        self.semantics = semantics
    
    def __call__(self, variables, wff, mask_vars=None, mask_fn=None, **kwargs):
        """
        mask_fn(mask_vars)
        """
        variables = [variables] if not isinstance(variables,(list,tuple)) else variables
        aggreg_doms = set([var.active_doms[0] for var in variables])
        if mask_fn is not None and mask_vars is not None:
            # create and apply the mask
            wff, mask = compute_mask(wff, mask_vars, mask_fn, aggreg_doms)
            ragged_wff = tf.ragged.boolean_mask(wff, mask)
            # aggregate
            aggreg_axes = [wff.active_doms.index(dom) for dom in aggreg_doms]
            result = self._aggreg_op(ragged_wff, aggreg_axes, **kwargs)
            if type(result) is tf.RaggedTensor:
                result = result.to_tensor()
            # For some values in the tensor, the mask can result in aggregating with empty variables.
            #    e.g. forall X ( exists Y:condition(X,Y) ( p(X,Y) ) )
            #       For some values of X, there may be no Y satisfying the condition
            # The result of the aggregation operator in such case is often not defined (e.g. nan).
            # We replace the result with 0.0 if the semantics of the aggregator is exists,
            # or 1.0 if the semantics of the aggregator is forall.
            aggreg_axes_in_mask = [mask.active_doms.index(dom) for dom in aggreg_doms 
                    if dom in mask.active_doms]
            non_empty_vars = tf.reduce_sum(tf.cast(mask,tf.int32), axis=aggreg_axes_in_mask) != 0
            empty_semantics = 1. if self.semantics == "forall" else 0
            result = tf.where(
                non_empty_vars,
                result,
                empty_semantics
            )
        else:
            aggreg_axes = [wff.active_doms.index(dom) for dom in aggreg_doms]
            result = self._aggreg_op(wff, axis=aggreg_axes, **kwargs)
        result.active_doms = [dom for dom in wff.active_doms if dom not in aggreg_doms]    
        undiag(*variables)
        return result

def compute_mask(wff, mask_vars, mask_fn, aggreg_doms):
    # 1. cross wff with args that are in the mask but not yet in the formula
    mask_vars_not_in_wff = [var for var in mask_vars if var.active_doms[0] not in wff.active_doms]
    wff = cross_args([wff]+mask_vars_not_in_wff)[0][0]
    # 2. set the masked vars on the first axes
    doms_in_mask = [var.active_doms[0] for var in mask_vars]
    doms_in_mask_not_aggreg = [dom for dom in doms_in_mask if dom not in aggreg_doms]
    doms_in_mask_aggreg = [dom for dom in doms_in_mask if dom in aggreg_doms]
    doms_not_in_mask = [dom for dom in wff.active_doms if dom not in doms_in_mask]
    new_doms_order = doms_in_mask_not_aggreg + doms_in_mask_aggreg + doms_not_in_mask
    wff = transpose_doms(wff, new_doms_order)
    # 3. compute the boolean mask from the masked vars
    crossed_mask_vars, doms_order_in_mask, dims0 = cross_args(mask_vars, flatten_dim0=True)
    mask = mask_fn(crossed_mask_vars)
    mask = tf.reshape(mask, dims0)
    # 4. shape it according to the var order in wff
    mask.active_doms = doms_order_in_mask
    mask = transpose_doms(mask, doms_in_mask_not_aggreg + doms_in_mask_aggreg)
    return wff, mask
    
def transpose_doms(wff, new_doms_order):
    perm = [wff.active_doms.index(dom) for dom in new_doms_order]
    wff = tf.transpose(wff, perm)
    wff.active_doms = new_doms_order
    return wff