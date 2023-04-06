from warnings import warn

import tensorflow as tf


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