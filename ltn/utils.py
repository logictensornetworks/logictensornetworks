from warnings import warn

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

def Pred_equal_smooth_exp(alpha=1):
    """
    Returns a 2-ary LTN predicate. It returns exp(-alpha*d(u,v)), where d(u,v) is the
    Euclidean distance between u and v. 
    """
    return ltn.Predicate.Lambda(
            lambda args: tf.exp(-alpha*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))
    )

def Pred_equal_smooth_inv(alpha=1):
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

class LogitsToPredicateModel(tf.keras.Model):
    """
    Given a model C that outputs k logits (for k classes)
        e.g. C(x) returns k values, not bounded in [0,1]
    `Cp = LogitsToPredicateModel(C)` is a corresponding model that returns
    probabilities for the class at the given index.
        e.g. Cp([x,i]) where i=0,1,...,k-1, returns 1 value in [0,1] for class i
    """
    def __init__(self, logits_model, single_label=True):
        """
        logits_model: a tf Model that outputs logits
        single_label: True for exclusive classes (logits are translated into probabilities using softmax),
                False for non-exclusive classes (logits are translated into probabilities using sigmoid)
        """
        warn("`LogitsToPredicateModel` is deprecated. " 
             "Use `ltn.Predicate.FromLogits` instead.", DeprecationWarning, stacklevel=2)
        super(LogitsToPredicateModel, self).__init__()
        self.logits_model = logits_model
        self.to_probs = tf.nn.softmax if single_label else tf.math.sigmoid

    def call(self, inputs):
        """
        inputs[0] are the inputs to the logits_model, for which we compute probabilities.
            probs[i] = to_probs(logits_model(inputs[0,i]))
        inputs[1] are the classes to index such that:
            results[i] = probs[i,inputs[1][i]]
        """
        x = inputs[0]
        logits = self.logits_model(x)
        probs = self.to_probs(logits)
        indices = tf.stack(inputs[1:], axis=1)
        indices = tf.cast(indices, tf.int32)
        return tf.gather_nd(probs, indices, batch_dims=1)