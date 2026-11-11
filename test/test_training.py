"""Small end-to-end learning tasks, to catch breakage across TensorFlow/Keras versions."""

import unittest

import numpy as np
import tensorflow as tf

import ltn


class TestBinaryClassification(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(0)
        points = np.random.uniform(size=(100, 2)).astype(np.float32)
        self.labels = np.linalg.norm(points - 0.5, axis=1) < 0.3
        self.x_pos = ltn.Variable("x_pos", points[self.labels])
        self.x_neg = ltn.Variable("x_neg", points[~self.labels])
        self.x_all = ltn.Variable("x_all", points)

        self.A = ltn.Predicate.MLP([2], hidden_layer_sizes=(16, 16))
        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.Forall = ltn.Wrapper_Quantifier(
            ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall"
        )
        self.SatAgg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def axioms(self):
        return self.SatAgg(
            [
                self.Forall(self.x_pos, self.A(self.x_pos)),
                self.Forall(self.x_neg, self.Not(self.A(self.x_neg))),
            ]
        ).tensor

    def train(self, steps, step_fn):
        for _ in range(steps):
            step_fn()

    def eager_step(self):
        with tf.GradientTape() as tape:
            loss = 1.0 - self.axioms()
        grads = tape.gradient(loss, self.A.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.A.trainable_variables))

    def accuracy(self):
        predictions = self.A(self.x_all).tensor.numpy() > 0.5
        return np.mean(predictions == self.labels)

    def test_eager(self):
        sat_before = float(self.axioms())
        self.train(300, self.eager_step)
        sat_after = float(self.axioms())
        self.assertGreater(sat_after, sat_before + 0.2)
        self.assertGreater(sat_after, 0.7)
        self.assertGreater(self.accuracy(), 0.9)

    def test_tf_function(self):
        step = tf.function(self.eager_step)
        sat_before = float(self.axioms())
        self.train(300, step)
        self.assertGreater(float(self.axioms()), sat_before + 0.2)
        self.assertGreater(self.accuracy(), 0.9)


class TestRelationalLearning(unittest.TestCase):
    """Learn a predicate and a function jointly, with a guarded quantifier and diag."""

    def setUp(self):
        tf.keras.utils.set_random_seed(0)
        self.points = np.random.uniform(-1.0, 1.0, size=(30, 1)).astype(np.float32)
        self.x = ltn.Variable("x", self.points)
        self.y = ltn.Variable("y", self.points)
        # f should learn to double its input
        self.targets = ltn.Variable("targets", 2 * self.points)
        self.f = ltn.Function.MLP(input_shapes=[1], output_shape=[1], hidden_layer_sizes=(16,))
        # Smaller(x,y) should learn x < y
        self.Smaller = ltn.Predicate.MLP([1, 1], hidden_layer_sizes=(16,))
        self.Eq = ltn.Predicate.Lambda(
            lambda args: tf.exp(-tf.reduce_sum(tf.abs(args[0] - args[1]), axis=1))
        )
        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.Forall = ltn.Wrapper_Quantifier(
            ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall"
        )
        self.SatAgg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))
        self.mask = ltn.Predicate.Lambda(lambda args: args[0][:, 0] < args[1][:, 0])(
            [self.x, self.y]
        )
        self.trainable_variables = self.f.trainable_variables + self.Smaller.trainable_variables
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def axioms(self):
        x, y = self.x, self.y
        return self.SatAgg(
            [
                self.Forall(ltn.diag(x, self.targets), self.Eq([self.f(x), self.targets])),
                self.Forall([x, y], self.Smaller([x, y]), mask=self.mask),
                self.Forall([x, y], self.Not(self.Smaller([y, x])), mask=self.mask),
            ]
        ).tensor

    def test_learns(self):
        sat_before = float(self.axioms())

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                loss = 1.0 - self.axioms()
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        for _ in range(300):
            step()
        sat_after = float(self.axioms())
        self.assertGreater(sat_after, sat_before + 0.2)
        self.assertGreater(sat_after, 0.7)
        # diag was undone by the quantifier
        self.assertEqual(self.x.free_vars, ["x"])
        predictions = self.f(self.x).tensor.numpy()
        self.assertLess(np.mean(np.abs(predictions - 2 * self.points)), 0.2)


class TestEmbeddingLearning(unittest.TestCase):
    """Learn embeddings and a relation jointly, as the parent_ancestor example does.

    The optimizer gets the model weights first, then the plain `tf.Variable`s of the trainable
    constants. Keras 3.3 to 3.6 crashed on that mix (issue #33).
    """

    def setUp(self):
        tf.keras.utils.set_random_seed(0)
        self.individuals = [
            ltn.Constant(np.random.uniform(size=4), trainable=True) for _ in range(5)
        ]
        self.parents = [(0, 1), (1, 2), (2, 3), (3, 4)]
        self.Parent = ltn.Predicate.MLP([4, 4], hidden_layer_sizes=(8, 8))
        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.Forall = ltn.Wrapper_Quantifier(
            ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall"
        )
        self.SatAgg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))
        self.trainable_variables = self.Parent.trainable_variables + ltn.as_tensors(
            self.individuals
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def axioms(self, tape):
        g = self.individuals
        x = ltn.Variable.from_constants("x", g, tape=tape)
        return self.SatAgg(
            [self.Parent([g[i], g[j]]) for i, j in self.parents]
            + [self.Not(self.Parent([g[j], g[i]])) for i, j in self.parents]
            + [self.Forall(x, self.Not(self.Parent([x, x])))]
        ).tensor

    def gradients(self):
        with tf.GradientTape() as tape:
            sat = self.axioms(tape)
            loss = 1.0 - sat
        return sat, tape.gradient(loss, self.trainable_variables)

    def train_and_check(self, step):
        values_before = [variable.numpy() for variable in self.trainable_variables]
        sat_before = float(step())
        for _ in range(300):
            sat_after = float(step())
        self.assertGreater(sat_after, sat_before + 0.2)
        self.assertGreater(sat_after, 0.7)
        for variable, value_before in zip(self.trainable_variables, values_before):
            self.assertFalse(np.allclose(variable.numpy(), value_before))
        g = self.individuals
        for i, j in self.parents:
            self.assertGreater(float(self.Parent([g[i], g[j]]).tensor), 0.5)
            self.assertLess(float(self.Parent([g[j], g[i]]).tensor), 0.5)

    def test_optimizer_outside_tf_function(self):
        """As in the examples: only the axioms run in a `tf.function`."""
        gradients = tf.function(self.gradients)

        def step():
            sat, grads = gradients()
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return sat

        self.train_and_check(step)

    def test_optimizer_inside_tf_function(self):
        @tf.function
        def step():
            sat, grads = self.gradients()
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return sat

        self.train_and_check(step)


if __name__ == "__main__":
    unittest.main()
