from typing import Optional, Union, Callable, Any

import tensorflow as tf

"""
Element-wise fuzzy logic operators for tensorflow.
Supports traditional NumPy/Tensorflow broadcasting.

To use in LTN formulas (broadcasting w.r.t. ltn variables appearing in a formula), 
wrap the operator with `ltn.WrapperConnective` or `ltn.WrapperQuantifier`. 
"""

eps = 1e-4
not_zeros = lambda x: (1-eps)*x + eps
not_ones = lambda x: (1-eps)*x

class And_Min:
    def __call__(self, x:tf.Tensor, y:tf.Tensor) -> tf.Tensor:
        return tf.minimum(x,y)

class And_Sum:
    def __call__(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return x+y

class Or_Max:
    def __call__(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.maximum(x,y)

class Or_LogMeanExp:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: tf.Tensor, y: tf.Tensor, alpha: Optional[float] = None)-> tf.Tensor:
        alpha = self.alpha if alpha is None else alpha 
        return reduce_logmeanexp(tf.stack([x,y],axis=-1)*alpha, axis=-1)/alpha

class Or_LogSumExp:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: tf.Tensor, y: tf.Tensor, alpha: Optional[float] = None)-> tf.Tensor:
        alpha = self.alpha if alpha is None else alpha 
        return tf.reduce_logsumexp(tf.stack([x,y],axis=-1)*alpha, axis=-1)/alpha

class Aggreg_Min:
    def __call__(self, xs: tf.Tensor, axis: Optional[Union[int, list[int]]]=None, keepdims: bool=False) -> tf.Tensor:
        return tf.reduce_min(xs,axis=axis,keepdims=keepdims)
class Aggreg_Max:
    def __call__(self, xs: tf.Tensor, axis: Optional[Union[int, list[int]]]=None, keepdims: bool=False) -> tf.Tensor:
        return tf.reduce_max(xs,axis=axis,keepdims=keepdims)
class Aggreg_Mean:
    def __call__(self, xs: tf.Tensor, axis: Optional[Union[int, list[int]]]=None, keepdims: bool=False) -> tf.Tensor:
        return tf.reduce_mean(xs,axis=axis,keepdims=keepdims)
class Aggreg_Sum:
    def __call__(self, xs: tf.Tensor, axis: Optional[Union[int, list[int]]]=None, keepdims: bool=False) -> tf.Tensor:
        return tf.reduce_sum(xs,axis=axis,keepdims=keepdims)

class Aggreg_LogMeanExp:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, xs: tf.Tensor, axis: Optional[Union[int,list[int]]] = None, keepdims: bool = False, 
                alpha: Optional[float] = None) -> tf.Tensor:
        alpha = self.alpha if alpha is None else alpha 
        return reduce_logmeanexp(xs*alpha, axis=axis, keepdims=keepdims)/alpha

class Aggreg_LogSumExp:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, xs: tf.Tensor, axis: Optional[Union[int,list[int]]] = None, keepdims: bool = False, 
                alpha: Optional[float] = None) -> tf.Tensor:
        alpha = self.alpha if alpha is None else alpha 
        return tf.reduce_logsumexp(xs*alpha, axis=axis, keepdims=keepdims)/alpha



def reduce_logmeanexp(xs: tf.Tensor, axis: Optional[Union[int, list[int]]]=None, keepdims: bool=False):
    """
    The added factor 1/n, by taking a mean instead of sum,
    is equivalent to approaching the max from a lower bound:
    https://en.wikipedia.org/wiki/LogSumExp
        max(X) < 1/t LSE(tX) <= max(X)+log(n)/t
    <=> max(X) - log(n)/t < 1/t LSE(tX) - log(n)/t <= max(X)
    <=> max(X) - log(n)/t < 1/t (LSE(tX)-log(n)) <= max(X)
    and then the n can enter in the log as a division.
    This is useful in our log context because we want all values to remain negative.

    We also do the stabilisation trick of removing the max value.
    """
    max_xs_raw = tf.stop_gradient(tf.reduce_max(xs, axis=axis,keepdims=True))
    max_xs_sq = tf.squeeze(max_xs_raw, axis=axis)
    return max_xs_sq+tf.math.log(tf.reduce_mean(tf.math.exp((xs-max_xs_raw)), axis=axis, keepdims=keepdims))