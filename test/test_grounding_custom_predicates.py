"""Custom predicates keep their input convention and expose only trainable parameters."""

import unittest

import numpy as np
import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


class _ListLogits(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__(name="list_logits")
        self.dense = tf.keras.layers.Dense(1, kernel_initializer="ones", bias_initializer="zeros")

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        return self.dense(inputs[0])


def _list_grounding(role: str = "axiom") -> Grounding:
    problem = tptp.parse(
        f"fof(rule,{role}, ! [X] : p(X)). fof(first,conjecture,p(a)). fof(second,conjecture,p(b))."
    )
    grounding = Grounding(problem, embedding_size=1)
    grounding.constants["a"] = ltn.Constant([1.0], trainable=False)
    grounding.constants["b"] = ltn.Constant([2.0], trainable=False)
    grounding.predicates["p"] = ltn.Predicate.FromLogits(
        _ListLogits(), activation_function="sigmoid", with_class_indexing=False
    )
    return grounding


class TestCustomUnaryPredicates(unittest.TestCase):
    def test_from_logits_matches_direct_eager_and_compiled_evaluation(self) -> None:
        grounding = _list_grounding()
        predicate = grounding.predicates["p"]
        x = ltn.Variable("X", [[1.0], [2.0]])
        quantified = grounding.quantifiers["!"](x, predicate([x]))
        expected = [
            float(predicate([grounding.constants["a"]]).tensor),
            float(quantified.tensor),
            float(grounding.aggregator([quantified]).tensor),
        ]

        def evaluate() -> tf.Tensor:
            return tf.stack(
                [
                    grounding.evaluate("p(a)").tensor,
                    grounding.evaluate("! [X] : p(X)").tensor,
                    grounding.sat(),
                ]
            )

        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                compute = tf.function(evaluate) if compiled else evaluate
                np.testing.assert_allclose(compute().numpy(), expected, rtol=1e-6)

    def test_from_logits_can_be_trained_and_refuted(self) -> None:
        query = "! [X] : p(X)"
        for operation in ("train", "refute"):
            with self.subTest(operation=operation):
                grounding = _list_grounding("axiom" if operation == "train" else "conjecture")
                before = float(grounding.evaluate(query).tensor)
                optimizer = tf.keras.optimizers.SGD(0.1)
                if operation == "train":
                    grounding.train(1, optimizer=optimizer)
                    self.assertGreater(float(grounding.evaluate(query).tensor), before)
                else:
                    satisfaction, truth = grounding.refute(query, steps=1, optimizer=optimizer)
                    self.assertEqual(satisfaction, 1.0)
                    self.assertLess(truth, before)
                    self.assertAlmostEqual(truth, float(grounding.evaluate(query).tensor))
                self.assertEqual(int(optimizer.iterations), 1)

    def test_ordinary_unary_predicates_and_functions_still_receive_tensors(self) -> None:
        grounding = Grounding(tptp.parse("fof(fact,axiom,p(f(a)))."), embedding_size=1)
        grounding.constants["a"] = ltn.Constant([1.0], trainable=False)
        grounding.functions["f"] = ltn.Function.Lambda(lambda x: x * 2.0)
        grounding.predicates["p"] = ltn.Predicate.Lambda(lambda x: tf.math.sigmoid(x[:, 0]))
        direct = grounding.predicates["p"](grounding.functions["f"](grounding.constants["a"]))
        self.assertAlmostEqual(float(grounding.evaluate("p(f(a))").tensor), float(direct.tensor))
        self.assertAlmostEqual(
            float(tf.function(grounding.sat)()),
            float(grounding.aggregator([direct]).tensor),
        )


class TestCustomTrainingParameters(unittest.TestCase):
    def test_fixed_constants_and_propositions_work_in_standard_optimizer_loops(self) -> None:
        for build_first in (False, True):
            with self.subTest(build_first=build_first):
                grounding = Grounding(tptp.parse("fof(fact,axiom,p(a) & s & t)."), embedding_size=1)
                grounding.constants["a"] = ltn.Constant([1.0], trainable=False)
                grounding.propositions["s"] = ltn.Proposition(0.8, trainable=False)
                model = tf.keras.Sequential(
                    [
                        tf.keras.Input(shape=(1,)),
                        tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros"),
                    ]
                )
                grounding.predicates["p"] = ltn.Predicate(model)
                parameters = grounding.trainable_variables
                expected = [grounding.propositions["t"].tensor, *model.trainable_variables]
                self.assertEqual({id(p) for p in parameters}, {id(p) for p in expected})
                self.assertTrue(all(p.trainable for p in parameters))
                optimizer = tf.keras.optimizers.Adam(0.01)
                if build_first:
                    optimizer.build(parameters)

                before = float(grounding.sat())
                with tf.GradientTape() as tape:
                    loss = 1.0 - grounding.sat(tape=tape, training=True)
                gradients = tape.gradient(loss, parameters)
                optimizer.apply_gradients(zip(gradients, parameters, strict=True))

                self.assertGreater(float(grounding.sat()), before)
                self.assertGreater(float(grounding.propositions["t"].tensor), 0.5)
                self.assertTrue(all(np.all(weight.numpy() > 0.0) for weight in model.weights))
                np.testing.assert_array_equal(grounding.constants["a"].tensor.numpy(), [1.0])
                self.assertAlmostEqual(float(grounding.propositions["s"].tensor), 0.8)
