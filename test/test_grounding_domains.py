"""Herbrand domains are evaluated only when quantification needs them."""

import unittest

import numpy as np
import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


def _grounding(axioms: str) -> tuple[Grounding, tf.keras.layers.BatchNormalization]:
    grounding = Grounding(
        tptp.parse(axioms + " fof(goal, conjecture, p(f(a)))."),
        embedding_size=1,
        herbrand_depth=1,
    )
    grounding.constants["a"] = ltn.Constant([2.0], trainable=False)
    norm = tf.keras.layers.BatchNormalization(momentum=0.5)
    model = tf.keras.Sequential([tf.keras.Input(shape=(1,)), norm])
    grounding.functions["f"] = ltn.Function(model)
    grounding.predicates["p"] = ltn.Predicate.Lambda(lambda x: tf.math.sigmoid(x[:, 0]))
    return grounding, norm


class TestGroundingDomains(unittest.TestCase):
    def test_implicit_distinctness_still_validates_its_constants(self) -> None:
        grounding = Grounding(tptp.parse("fof(goal, conjecture, p(1) & p(2))."))
        del grounding.constants["1"]

        with self.assertRaisesRegex(ValueError, "constant 1 has no grounding"):
            grounding.sat()

    def test_unused_herbrand_functions_do_not_update_state(self) -> None:
        for axioms in ("", "fof(fact, axiom, s).", "fof(fact, axiom, p(a))."):
            with self.subTest(axioms=axioms):
                grounding, norm = _grounding(axioms)
                expected = float(grounding.sat())
                self.assertAlmostEqual(float(grounding.sat(training=True)), expected)
                self.assertAlmostEqual(grounding.train(1, tf.keras.optimizers.SGD(0.0)), expected)
                self.assertAlmostEqual(
                    float(grounding.evaluate("p(a)", training=True).tensor),
                    float(tf.math.sigmoid(2.0)),
                )
                self.assertEqual(float(grounding.evaluate("2 = 2", training=True).tensor), 1.0)
                np.testing.assert_array_equal(norm.moving_mean.numpy(), [0.0])

                # An explicitly used function still executes, without an extra domain call.
                self.assertAlmostEqual(
                    float(grounding.evaluate("p(f(a))", training=True).tensor), 0.5
                )
                np.testing.assert_array_equal(norm.moving_mean.numpy(), [1.0])

    def test_quantified_axioms_share_a_domain_and_keep_its_gradients(self) -> None:
        grounding, norm = _grounding(
            "fof(all_p, axiom, ! [X] : p(X)). fof(some_q, axiom, ? [Y] : q(Y))."
        )
        grounding.predicates["q"] = ltn.Predicate.Lambda(lambda x: tf.math.sigmoid(x[:, 0] - 1.0))
        with tf.GradientTape() as tape:
            actual = grounding.sat(tape=tape, training=True)
        gradient = tape.gradient(actual, norm.beta)

        # The single-element batch normalizes f(a) to beta. Construct that domain directly
        # to check both its truth and the useful gradient through the generated individual.
        with tf.GradientTape() as tape:
            individuals = ltn.Variable(
                "Expected", tf.stack([grounding.constants["a"].tensor, norm.beta])
            )
            expected = grounding.aggregator(
                [
                    grounding.quantifiers["!"](individuals, grounding.predicates["p"](individuals)),
                    grounding.quantifiers["?"](individuals, grounding.predicates["q"](individuals)),
                ]
            ).tensor
        reference = tape.gradient(expected, norm.beta)
        self.assertAlmostEqual(float(actual), float(expected))
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient[0]), 1e-4)
        np.testing.assert_allclose(gradient.numpy(), reference.numpy(), rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(norm.moving_mean.numpy(), [1.0])

        result = grounding.train(1, tf.keras.optimizers.SGD(0.1))
        self.assertAlmostEqual(result, float(expected))
        np.testing.assert_allclose(norm.beta.numpy(), 0.1 * reference.numpy(), rtol=1e-5, atol=1e-7)
        # One more domain evaluation, shared by both formulas, updates 1.0 to 1.5.
        np.testing.assert_array_equal(norm.moving_mean.numpy(), [1.5])
        grounding.sat()
        np.testing.assert_array_equal(norm.moving_mean.numpy(), [1.5])

    def test_explicit_individuals_override_herbrand_expansion(self) -> None:
        grounding, norm = _grounding("")
        with self.assertRaises(ValueError):
            grounding.evaluate("! [X] : p(X)", individuals=[], training=True)
        self.assertEqual(
            float(grounding.evaluate("! [X] : $true", individuals=[], training=True).tensor), 1.0
        )
        override = [ltn.Constant([4.0], trainable=False)]
        actual = grounding.evaluate("! [X] : p(X)", individuals=override, training=True)
        variable = ltn.Variable("Expected", [[4.0]])
        expected = grounding.quantifiers["!"](variable, grounding.predicates["p"](variable))
        self.assertAlmostEqual(float(actual.tensor), float(expected.tensor))
        np.testing.assert_array_equal(norm.moving_mean.numpy(), [0.0])
