"""Implicit distinct-value axioms obey the same groundings as explicit formulas."""

import itertools
import unittest

import numpy as np
import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


def _grounding(names: tuple[str, ...]) -> Grounding:
    formula = " & ".join(f"{left} != {right}" for left, right in itertools.pairwise(names))
    grounding = Grounding(tptp.parse(f"fof(fact, axiom, {formula})."), embedding_size=1)
    for index, name in enumerate(names):
        grounding.constants[name].tensor.assign([index * 0.75])
    return grounding


def _manual_sat(grounding: Grounding, names: tuple[str, ...]) -> tf.Tensor:
    # Write each inequality with ordinary LTN objects, independently of the parser's
    # batched pair construction. Equal-value aliases and self-pairs must not be counted.
    inequalities = [
        grounding.Not(grounding.predicates["="]([grounding.constants[a], grounding.constants[b]]))
        for a, b in itertools.permutations(names, 2)
    ]
    forall = ltn.Wrapper_Formula_Aggregator(grounding.quantifiers["!"].aggreg_op)
    axioms = [grounding.evaluate(statement.formula) for statement in grounding.problem.statements]
    return grounding.aggregator([*axioms, forall(inequalities)]).tensor


class _Equality(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__(name="custom_equality")
        self.scale = self.add_weight(
            name="scale", shape=(), initializer=tf.keras.initializers.Constant(0.75)
        )

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        distance = tf.reduce_sum(tf.square(inputs[0] - inputs[1]), axis=1)
        return tf.cast(tf.exp(-tf.nn.softplus(self.scale) * distance), tf.float64)


class TestDistinctValueGroundings(unittest.TestCase):
    def test_custom_equality_dtypes_work_in_eager_and_compiled_operations(self) -> None:
        for names in (("1", "2", "3"), ('"Apple"', '"Pear"', '"Plum"')):
            for dtype in (tf.bool, tf.float64):
                with self.subTest(names=names, dtype=dtype.name):
                    grounding = _grounding(names)
                    if dtype == tf.bool:
                        equality = lambda args: tf.reduce_all(tf.equal(args[0], args[1]), axis=1)
                    else:
                        equality = lambda args: tf.cast(
                            tf.exp(-tf.reduce_sum(tf.square(args[0] - args[1]), axis=1)),
                            tf.float64,
                        )
                    grounding.predicates["="] = ltn.Predicate.Lambda(equality)
                    expected = float(_manual_sat(grounding, names))

                    for compute in (grounding.sat, tf.function(grounding.sat)):
                        actual = compute()
                        self.assertEqual(actual.dtype, tf.float32)
                        self.assertEqual(actual.shape, ())
                        self.assertAlmostEqual(float(actual), expected, places=6)
                    self.assertAlmostEqual(
                        grounding.train(1, tf.keras.optimizers.SGD(0.0)), expected, places=6
                    )
                    sat, truth = grounding.refute(
                        "$true", steps=1, optimizer=tf.keras.optimizers.SGD(0.0)
                    )
                    self.assertAlmostEqual(sat, expected, places=6)
                    self.assertEqual(truth, 1.0)

    def test_implicit_inequalities_use_configured_negation(self) -> None:
        names = ("1", "2")
        grounding = _grounding(names)
        grounding.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Godel())
        expected = float(_manual_sat(grounding, names))
        self.assertAlmostEqual(expected, 1.0 - np.sqrt(0.5), places=6)
        self.assertAlmostEqual(float(grounding.sat()), expected, places=6)
        self.assertAlmostEqual(float(tf.function(grounding.sat)()), expected, places=6)
        self.assertAlmostEqual(grounding.train(1, tf.keras.optimizers.SGD(0.0)), expected, places=6)
        sat, _ = grounding.refute("$true", steps=1, optimizer=tf.keras.optimizers.SGD(0.0))
        self.assertAlmostEqual(sat, expected, places=6)

    def test_custom_inequality_gradients_and_training_match_ltn(self) -> None:
        names = ("1", "2", "3")
        grounding = _grounding(names)
        grounding.predicates["="] = ltn.Predicate(_Equality())
        grounding.Not = ltn.Wrapper_Connective(lambda truth: (1.0 - truth) / (1.0 + truth))

        with tf.GradientTape(persistent=True) as tape:
            actual = grounding.sat(tape=tape)
            expected = _manual_sat(grounding, names)
        parameters = grounding.trainable_variables
        gradients = tape.gradient(actual, parameters)
        expected_gradients = tape.gradient(expected, parameters)
        self.assertAlmostEqual(float(actual), float(expected), places=6)
        for gradient, reference in zip(gradients, expected_gradients, strict=True):
            self.assertIsNotNone(gradient)
            self.assertIsNotNone(reference)
            np.testing.assert_allclose(gradient.numpy(), reference.numpy(), rtol=1e-5, atol=1e-7)
        # The end embeddings and equality model must all receive useful gradients.
        for index in (0, 2, 3):
            self.assertGreater(float(tf.reduce_max(tf.abs(gradients[index]))), 1e-4)

        before = [parameter.numpy().copy() for parameter in parameters]
        grounding.train(1, tf.keras.optimizers.SGD(0.01), reuse_graph=True)
        for parameter, start, gradient in zip(parameters, before, expected_gradients, strict=True):
            np.testing.assert_allclose(
                parameter.numpy(), start + 0.01 * gradient.numpy(), rtol=1e-5, atol=1e-7
            )
        self.assertGreater(float(grounding.sat()), float(actual))
