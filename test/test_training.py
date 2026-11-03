"""Small end-to-end learning tasks, to catch breakage across TensorFlow/Keras versions."""
import unittest

import numpy as np
import tensorflow as tf

import ltn


class TestBinaryClassification(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(0)
        points = np.random.uniform(size=(100,2)).astype(np.float32)
        self.labels = np.linalg.norm(points - .5, axis=1) < .3
        self.x_pos = ltn.Variable('x_pos', points[self.labels])
        self.x_neg = ltn.Variable('x_neg', points[~self.labels])
        self.x_all = ltn.Variable('x_all', points)

        self.A = ltn.Predicate.MLP([2], hidden_layer_sizes=(16,16))
        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
        self.SatAgg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

    def axioms(self):
        return self.SatAgg([
            self.Forall(self.x_pos, self.A(self.x_pos)),
            self.Forall(self.x_neg, self.Not(self.A(self.x_neg))),
        ]).tensor

    def train(self, steps, step_fn):
        for _ in range(steps):
            step_fn()

    def eager_step(self):
        with tf.GradientTape() as tape:
            loss = 1. - self.axioms()
        grads = tape.gradient(loss, self.A.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.A.trainable_variables))

    def accuracy(self):
        predictions = self.A(self.x_all).tensor.numpy() > .5
        return np.mean(predictions == self.labels)

    def test_eager(self):
        sat_before = float(self.axioms())
        self.train(300, self.eager_step)
        sat_after = float(self.axioms())
        self.assertGreater(sat_after, sat_before + .2)
        self.assertGreater(sat_after, .7)
        self.assertGreater(self.accuracy(), .9)

    def test_tf_function(self):
        step = tf.function(self.eager_step)
        sat_before = float(self.axioms())
        self.train(300, step)
        self.assertGreater(float(self.axioms()), sat_before + .2)
        self.assertGreater(self.accuracy(), .9)


class TestRelationalLearning(unittest.TestCase):
    """Learn a predicate and a function jointly, with a guarded quantifier and diag."""

    def setUp(self):
        tf.keras.utils.set_random_seed(0)
        self.points = np.random.uniform(-1., 1., size=(30,1)).astype(np.float32)
        self.x = ltn.Variable('x', self.points)
        self.y = ltn.Variable('y', self.points)
        # f should learn to double its input
        self.targets = ltn.Variable('targets', 2*self.points)
        self.f = ltn.Function.MLP(input_shapes=[1], output_shape=[1], hidden_layer_sizes=(16,))
        # Smaller(x,y) should learn x < y
        self.Smaller = ltn.Predicate.MLP([1,1], hidden_layer_sizes=(16,))
        self.Eq = ltn.Predicate.Lambda(lambda args: tf.exp(-tf.reduce_sum(tf.abs(args[0]-args[1]),axis=1)))
        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
        self.SatAgg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))
        self.mask = ltn.Predicate.Lambda(lambda args: args[0][:,0] < args[1][:,0])([self.x, self.y])
        self.trainable_variables = self.f.trainable_variables + self.Smaller.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

    def axioms(self):
        x, y = self.x, self.y
        return self.SatAgg([
            self.Forall(ltn.diag(x, self.targets), self.Eq([self.f(x), self.targets])),
            self.Forall([x,y], self.Smaller([x,y]), mask=self.mask),
            self.Forall([x,y], self.Not(self.Smaller([y,x])), mask=self.mask),
        ]).tensor

    def test_learns(self):
        sat_before = float(self.axioms())

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                loss = 1. - self.axioms()
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        for _ in range(300):
            step()
        sat_after = float(self.axioms())
        self.assertGreater(sat_after, sat_before + .2)
        self.assertGreater(sat_after, .7)
        # diag was undone by the quantifier
        self.assertEqual(self.x.free_vars, ['x'])
        predictions = self.f(self.x).tensor.numpy()
        self.assertLess(np.mean(np.abs(predictions - 2*self.points)), .2)


if __name__ == "__main__":
    unittest.main()
