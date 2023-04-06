import tensorflow as tf

import ltn


def Pred_equal_strict():
    """Returns a 2-ary LTN predicate. The predicate returns 1.0 if the inputs are equal, 0.0 otherwise.
    """
    # The equality is measured on the list of tensors unpacked on dimension 0.
    # The model will return n truth values, where n is the number of values on dimension 0 (the batch dimension). 
    return ltn.Predicate.Lambda(
            lambda args: tf.cast(
                tf.reduce_all(tf.math.equal(args[0],args[1]),axis=tf.range(1,tf.rank(args[0]))),
                dtype=tf.float32
            ))


def Pred_equal_smooth_exp(alpha=1.):
    """
    Returns a 2-ary LTN predicate. It returns exp(-alpha*d(u,v)), where d(u,v) is the
    Euclidean distance between u and v. 
    """
    return ltn.Predicate.Lambda(
            lambda args: tf.exp(-alpha*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))
    )


def Pred_equal_smooth_inv(alpha=1.):
    """
    Returns a 2-ary LTN predicate. It returns 1/(1+alpha*d(u,v)), where d(u,v) is the
    Euclidean distance between u and v. 
    """
    return ltn.Predicate.Lambda(
            lambda args: 1/(1+alpha*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))
    )


def Func_add():
    """Returns a 2-ary LTN function. The function returns the element-wise addition of the inputs.
    """
    return ltn.Function.Lambda(lambda args: args[0]+args[1])
