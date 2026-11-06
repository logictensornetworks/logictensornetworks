"""Regression coverage for model arguments, empty domains and term transposes."""

import unittest

import numpy as np
import tensorflow as tf

import ltn
from ltn import core, fuzzy_ops


class ParameterizedLogits(tf.keras.Model):
    def call(self, inputs, scale=1.0, offset=0.0):
        return inputs[0] * scale + offset


class TestFromLogitsArguments(unittest.TestCase):
    def test_custom_arguments_reach_logits_model(self):
        for activation, indexing in [("sigmoid", False), ("sigmoid", True), ("softmax", True)]:
            values = (
                tf.constant([[1.0, -2.0], [3.0, -4.0]]) if indexing else tf.constant([[1.0], [3.0]])
            )
            x = ltn.Variable("x", values)
            predicate = ltn.Predicate.FromLogits(
                ParameterizedLogits(), activation, with_class_indexing=indexing
            )
            inputs = [x, ltn.Variable("class", [0.0, 1.0])] if indexing else [x]

            def evaluate(scale, offset, positional=False):
                if positional:
                    return predicate(inputs, scale, offset=offset).tensor
                return predicate(inputs, scale=scale, offset=offset).tensor

            graph_evaluate = tf.function(evaluate)
            for runner in [evaluate, graph_evaluate]:
                for positional in [False, True]:
                    with self.subTest(
                        activation=activation,
                        indexing=indexing,
                        graph=runner is graph_evaluate,
                        positional=positional,
                    ):
                        scale, offset = tf.constant(-2.0), tf.constant(0.7)
                        logits = values * scale + offset
                        expected = (
                            tf.nn.softmax(logits) if activation == "softmax" else tf.sigmoid(logits)
                        )
                        if not indexing:
                            expected = expected[:, 0]
                        np.testing.assert_allclose(
                            runner(scale, offset, positional), expected, atol=1e-6
                        )


class TestEmptyQuantifierDomains(unittest.TestCase):
    def test_empty_domain_uses_quantifier_semantics(self):
        x = ltn.Variable("x", tf.zeros([0, 1]))
        wff = core.Formula(tf.zeros([0]), ["x"])
        operators = [
            fuzzy_ops.Aggreg_Mean(),
            fuzzy_ops.Aggreg_Min(),
            fuzzy_ops.Aggreg_Max(),
            fuzzy_ops.Aggreg_pMean(),
            fuzzy_ops.Aggreg_pMeanError(),
            fuzzy_ops.Aggreg_Prod(),
        ]
        for semantics, expected in [("forall", 1.0), ("exists", 0.0)]:
            for operator in operators:
                with self.subTest(semantics=semantics, operator=type(operator).__name__):
                    result = ltn.Wrapper_Quantifier(operator, semantics)(x, wff)
                    self.assertEqual(result.free_vars, [])
                    self.assertEqual(result.tensor.shape, [])
                    self.assertEqual(float(result.tensor), expected)

    def test_partial_and_joint_quantification_with_dynamic_domains(self):
        x = ltn.Variable("x", tf.zeros([0, 1]))
        y = ltn.Variable("y", tf.zeros([3, 1]))
        for semantics, empty_value in [("forall", 1.0), ("exists", 0.0)]:
            for variables, remaining_vars in [([x], ["z", "y"]), ([x, y], ["z"])]:
                for keepdims in [False, True]:
                    quantifier = ltn.Wrapper_Quantifier(fuzzy_ops.Aggreg_Mean(), semantics)

                    def evaluate(values):
                        result = quantifier(
                            variables, core.Formula(values, ["z", "x", "y"]), keepdims=keepdims
                        )
                        self.assertEqual(result.free_vars, remaining_vars)
                        return result.tensor

                    graph_evaluate = tf.function(
                        evaluate, input_signature=[tf.TensorSpec([None, None, None], tf.float32)]
                    )
                    for runner in [evaluate, graph_evaluate]:
                        with self.subTest(
                            semantics=semantics,
                            remaining=remaining_vars,
                            keepdims=keepdims,
                            graph=runner is graph_evaluate,
                        ):
                            axes = [1] if len(variables) == 1 else [1, 2]
                            if keepdims:
                                expected_shape = [2, 1, 3] if len(variables) == 1 else [2, 1, 1]
                            else:
                                expected_shape = [2, 3] if len(variables) == 1 else [2]
                            np.testing.assert_array_equal(
                                runner(tf.zeros([2, 0, 3])), np.full(expected_shape, empty_value)
                            )
                            # The same graph must still aggregate non-empty batches normally.
                            values = tf.reshape(tf.linspace(0.0, 1.0, 12), [2, 2, 3])
                            np.testing.assert_allclose(
                                runner(values), tf.reduce_mean(values, axis=axes, keepdims=keepdims)
                            )
                            # An empty remaining domain should produce an empty result.
                            self.assertEqual(tf.size(runner(tf.zeros([0, 2, 3]))), 0)

    def test_empty_batch_has_finite_zero_gradients(self):
        predicate = ltn.Predicate.MLP([1], hidden_layer_sizes=())
        forall = ltn.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMeanError(), "forall")

        def evaluate(values):
            with tf.GradientTape() as tape:
                x = ltn.Variable("x", values)
                satisfaction = forall(x, predicate(x)).tensor
            return satisfaction, tape.gradient(satisfaction, predicate.trainable_variables)

        graph_evaluate = tf.function(
            evaluate, input_signature=[tf.TensorSpec([None, 1], tf.float32)]
        )
        for runner in [evaluate, graph_evaluate]:
            with self.subTest(graph=runner is graph_evaluate):
                satisfaction, gradients = runner(tf.zeros([0, 1]))
                self.assertEqual(float(satisfaction), 1.0)
                for gradient in gradients:
                    self.assertIsNotNone(gradient)
                    np.testing.assert_array_equal(gradient, tf.zeros_like(gradient))
                satisfaction, gradients = runner(tf.ones([2, 1]))
                self.assertTrue(np.isfinite(satisfaction))
                self.assertTrue(all(np.all(np.isfinite(g)) for g in gradients))
                self.assertTrue(any(np.any(g != 0.0) for g in gradients))


class TestTransposeTermFeatures(unittest.TestCase):
    def test_feature_axes_are_preserved(self):
        values = tf.reshape(tf.range(120, dtype=tf.float32), [2, 3, 4, 5])
        for in_place in [False, True]:
            with self.subTest(in_place=in_place):
                term = core.Term(values, ["x", "y"])
                result = core.transpose_free_vars(term, ["y", "x"], in_place=in_place)
                self.assertIsInstance(result, core.Term)
                self.assertEqual(result.free_vars, ["y", "x"])
                np.testing.assert_array_equal(result.tensor, tf.transpose(values, [1, 0, 2, 3]))
                if in_place:
                    self.assertIs(result, term)
                else:
                    self.assertIsNot(result, term)
                    self.assertEqual(term.free_vars, ["x", "y"])
                    np.testing.assert_array_equal(term.tensor, values)

    def test_variables_and_constants(self):
        for term in [
            ltn.Variable("x", [[1.0], [2.0]]),
            ltn.Constant([[1.0, 2.0]], trainable=False),
        ]:
            with self.subTest(term=type(term).__name__):
                result = core.transpose_free_vars(term, term.free_vars)
                np.testing.assert_array_equal(result.tensor, term.tensor)
                self.assertEqual(result.free_vars, term.free_vars)

    def test_transpose_in_tf_function_preserves_gradients(self):
        @tf.function(input_signature=[tf.TensorSpec([None, None, 4, 5], tf.float32)])
        def transpose(values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                result = core.transpose_free_vars(core.Term(values, ["x", "y"]), ["y", "x"])
                loss = tf.reduce_sum(tf.square(result.tensor))
            return result.tensor, tape.gradient(loss, values)

        values = tf.reshape(tf.range(120, dtype=tf.float32), [2, 3, 4, 5])
        result, gradient = transpose(values)
        np.testing.assert_array_equal(result, tf.transpose(values, [1, 0, 2, 3]))
        np.testing.assert_array_equal(gradient, 2 * values)
