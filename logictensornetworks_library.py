import tensorflow as tf

def equal_simple(x,y):
    return tf.exp(-tf.reduce_sum(tf.abs(x-y),1,keepdims=True))

default_equal_diameter = 1.0

def equal_euclidian(t_1,t_2,diameter=default_equal_diameter):
    delta = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(t_1,t_2)),1,keepdims=True))
    return tf.exp(-tf.divide(delta,diameter))
