import unittest
import warnings

import numpy as np
import tensorflow as tf

import ltn
from ltn import core

def array_allclose(actual, desired, atol=1e-6):
    actual, desired = np.asarray(actual), np.asarray(desired)
    return np.allclose(actual, desired, atol=atol) and actual.shape == desired.shape


class ListInputLogits(tf.keras.Model):
    """Logits model called with a list of inputs, as required by `Predicate.FromLogits`."""
    def __init__(self, n_classes):
        super().__init__()
        self.dense = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        return self.dense(inputs[0])


class TestFromLogits(unittest.TestCase):
    def setUp(self):
        self.n_classes = 3
        self.x = core.Variable('x', np.random.default_rng(0).normal(size=(5,2)))
        self.l = core.Variable('l', np.arange(self.n_classes)) # class indices
        self.logits_model = ListInputLogits(self.n_classes)
        self.logits = self.logits_model([self.x.tensor]).numpy() # also builds the weights

    def test_softmax(self):
        P = core.Predicate.FromLogits(self.logits_model, activation_function="softmax")
        phi = P([self.x, self.l])
        self.assertEqual(phi.free_vars, ['x','l'])
        self.assertTrue(array_allclose(phi.tensor, tf.nn.softmax(self.logits)))
        # the classes are exclusive
        self.assertTrue(array_allclose(tf.reduce_sum(phi.tensor, axis=1), np.ones(5)))

    def test_sigmoid_with_class_indexing(self):
        P = core.Predicate.FromLogits(self.logits_model, activation_function="sigmoid")
        phi = P([self.x, self.l])
        self.assertEqual(phi.free_vars, ['x','l'])
        self.assertTrue(array_allclose(phi.tensor, tf.math.sigmoid(self.logits)))

    def test_sigmoid_without_class_indexing(self):
        logits_model = ListInputLogits(1)
        logits = logits_model([self.x.tensor])
        P = core.Predicate.FromLogits(logits_model, activation_function="sigmoid", with_class_indexing=False)
        phi = P([self.x])
        self.assertEqual(phi.free_vars, ['x'])
        self.assertTrue(array_allclose(phi.tensor, tf.math.sigmoid(logits)[:,0]))

    def test_index_with_constant(self):
        P = core.Predicate.FromLogits(self.logits_model, activation_function="softmax")
        phi = P([self.x, core.Constant(1, trainable=False)])
        self.assertEqual(phi.free_vars, ['x'])
        self.assertTrue(array_allclose(phi.tensor, tf.nn.softmax(self.logits)[:,1]))

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            core.Predicate.FromLogits(self.logits_model, activation_function="relu")

    def test_exposes_logits_model_and_weights(self):
        for activation in ["sigmoid","softmax"]:
            with self.subTest(activation=activation):
                P = core.Predicate.FromLogits(self.logits_model, activation_function=activation)
                self.assertIs(P.logits_model, self.logits_model)
                self.assertEqual(
                    [id(v) for v in P.trainable_variables],
                    [id(v) for v in self.logits_model.trainable_variables])

    def test_in_tf_function(self):
        """Keras 3 rejected the default model names (leading underscore) as name scopes in a tf.function."""
        for activation in ["sigmoid","softmax"]:
            with self.subTest(activation=activation):
                P = core.Predicate.FromLogits(self.logits_model, activation_function=activation)
                @tf.function
                def truth_values():
                    return P([self.x, self.l]).tensor
                self.assertTrue(array_allclose(truth_values(), P([self.x, self.l]).tensor))

    def test_gradients_reach_logits_model(self):
        P = core.Predicate.FromLogits(self.logits_model, activation_function="softmax")
        with tf.GradientTape() as tape:
            out = tf.reduce_sum(P([self.x, self.l]).tensor * tf.constant([1.,0.,0.]))
        grads = tape.gradient(out, P.trainable_variables)
        self.assertTrue(all(g is not None and np.any(g.numpy() != 0.) for g in grads))


class TestTake(unittest.TestCase):
    def setUp(self):
        self.x = core.Variable('x', np.random.default_rng(0).normal(size=(3,2)))
        self.y = core.Variable('y', np.random.default_rng(1).normal(size=(4,2)))
        self.p = core.Predicate.Lambda(lambda inputs: tf.reduce_sum(inputs[0]*inputs[1], axis=1))
        self.phi = self.p([self.x, self.y])

    def test_single_index_removes_free_var(self):
        out = self.phi.take('y', 2)
        self.assertIsInstance(out, core.Formula)
        self.assertEqual(out.free_vars, ['x'])
        self.assertTrue(array_allclose(out.tensor, self.phi.tensor[:,2]))

    def test_list_of_indices_keeps_free_var(self):
        out = self.phi.take('x', [0,2])
        self.assertEqual(out.free_vars, ['x','y'])
        self.assertTrue(array_allclose(out.tensor, tf.gather(self.phi.tensor, [0,2], axis=0)))

    def test_does_not_modify_original(self):
        self.phi.take('x', 0)
        self.assertEqual(self.phi.free_vars, ['x','y'])
        self.assertEqual(self.phi.tensor.shape, [3,4])

    def test_on_terms_keeps_feature_dims(self):
        out = self.x.take('x', 1)
        self.assertIsInstance(out, core.Term)
        self.assertEqual(out.free_vars, [])
        self.assertTrue(array_allclose(out.tensor, self.x.tensor[1]))

    def test_unknown_free_var(self):
        with self.assertRaises(ValueError):
            self.phi.take('z', 0)

    def test_invalid_indices_rank(self):
        with self.assertRaises(ValueError):
            self.phi.take('x', [[0,1]])


class TestVariableValidation(unittest.TestCase):
    def test_reserved_labels(self):
        for label in ["diag_x", "diaglock_x", "_flat_x"]:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    core.Variable(label, [[1.],[2.]])

    def test_scalar_values(self):
        with self.assertRaises(ValueError):
            core.Variable('x', 1.)

    def test_quantifier_semantics(self):
        with self.assertRaises(ValueError):
            core.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_Mean(), semantics="all")


class TestUtils(unittest.TestCase):
    def test_pred_equal_strict(self):
        u = core.Variable('u', [[1.,2.],[3.,4.],[1.,2.]])
        v = core.Variable('v', [[1.,2.],[3.,5.]])
        Eq = ltn.utils.Pred_equal_strict()
        phi = Eq([u,v])
        self.assertEqual(phi.free_vars, ['u','v'])
        self.assertTrue(array_allclose(phi.tensor, [[1.,0.],[0.,0.],[1.,0.]]))

    def test_pred_equal_strict_on_higher_rank_features(self):
        u = core.Variable('u', np.ones((2,2,3)))
        v = core.Variable('v', np.stack([np.ones((2,3)), np.zeros((2,3))]))
        self.assertTrue(array_allclose(ltn.utils.Pred_equal_strict()([u,v]).tensor, [[1.,0.],[1.,0.]]))

    def test_smooth_equalities_decrease_with_distance(self):
        u = core.Variable('u', [[0.,0.]])
        v = core.Variable('v', [[0.,1.],[0.,2.],[0.,4.]])
        for pred in [ltn.utils.Pred_equal_smooth_exp(), ltn.utils.Pred_equal_smooth_inv()]:
            out = pred([u,v]).tensor.numpy()[0]
            self.assertTrue(np.all(np.diff(out) < 0.))
            self.assertTrue(np.all((out > 0.) & (out < 1.)))

    def test_func_add(self):
        x = core.Variable('x', [[1.,2.],[3.,4.]])
        y = core.Variable('y', [[10.,20.],[30.,40.],[50.,60.]])
        out = ltn.utils.Func_add()([x,y])
        self.assertIsInstance(out, core.Term)
        self.assertEqual(out.free_vars, ['x','y'])
        self.assertEqual(out.tensor.shape, [2,3,2])
        self.assertTrue(array_allclose(out.tensor[1,2], [53.,64.]))

    def test_logits_to_predicate_model_is_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            ltn.utils.LogitsToPredicateModel(tf.keras.layers.Dense(3))

    def test_logits_to_predicate_model_values(self):
        x = tf.constant(np.random.default_rng(0).normal(size=(4,2)), dtype=tf.float32)
        indices = tf.constant([0,2,1,2])
        for single_label, activation in [(True, tf.nn.softmax), (False, tf.math.sigmoid)]:
            with self.subTest(single_label=single_label):
                logits_model = tf.keras.layers.Dense(3)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    model = ltn.utils.LogitsToPredicateModel(logits_model, single_label=single_label)
                out = model([x, indices])
                expected = tf.gather(activation(logits_model(x)), indices, batch_dims=1)
                self.assertTrue(array_allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
