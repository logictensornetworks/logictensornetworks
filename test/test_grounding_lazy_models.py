"""Custom models must be built by the first forward pass before collecting their weights."""

import unittest

import numpy as np
import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


def lazy_grounding(symbol: str, role: str = "axiom") -> tuple[Grounding, tf.keras.Model, str]:
    formula = "p(a)" if symbol == "predicate" else "p(f(a))"
    grounding = Grounding(tptp.parse(f"fof(test, {role}, {formula})."), embedding_size=1)
    grounding.constants["a"] = ltn.Constant([1.0], trainable=False)
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")]
    )
    if symbol == "predicate":
        grounding.predicates["p"] = ltn.Predicate.FromLogits(
            model, activation_function="sigmoid", with_class_indexing=False
        )
    else:
        grounding.functions["f"] = ltn.Function(model)
        grounding.predicates["p"] = ltn.Predicate.Lambda(lambda x: tf.math.sigmoid(x[:, 0]))
    return grounding, model, formula


@pytest.mark.parametrize("symbol", ["predicate", "function"])
def test_first_training_call_updates_an_initially_unbuilt_model(symbol: str) -> None:
    grounding, model, formula = lazy_grounding(symbol)
    assert not model.built

    grounding.train(1, optimizer=tf.keras.optimizers.SGD(0.1))

    assert all(np.all(weight.numpy() > 0.0) for weight in model.trainable_variables)
    assert float(grounding.evaluate(formula).tensor) > 0.5


@pytest.mark.parametrize("symbol", ["predicate", "function"])
def test_refutation_updates_a_model_built_only_by_the_query(symbol: str) -> None:
    grounding, model, formula = lazy_grounding(symbol, role="conjecture")
    assert not model.built

    grounding.refute(formula, steps=1, optimizer=tf.keras.optimizers.SGD(0.1))

    assert all(np.all(weight.numpy() < 0.0) for weight in model.trainable_variables)
    assert float(grounding.evaluate(formula).tensor) < 0.5


def test_kept_graph_keeps_training_a_model_built_on_its_first_call() -> None:
    grounding, model, formula = lazy_grounding("predicate")
    optimizer = tf.keras.optimizers.SGD(0.1)
    grounding.train(1, optimizer=optimizer, reuse_graph=True)
    first = [weight.numpy().copy() for weight in model.trainable_variables]
    first_truth = float(grounding.evaluate(formula).tensor)

    grounding.train(1, reuse_graph=True)

    assert first_truth > 0.5
    assert all(
        np.all(weight.numpy() > previous)
        for weight, previous in zip(model.trainable_variables, first, strict=True)
    )
    assert float(grounding.evaluate(formula).tensor) > first_truth
    assert int(optimizer.iterations) == 2


def test_unused_lazy_models_do_not_prevent_training_the_axioms() -> None:
    problem = tptp.parse("fof(fact, axiom, s). fof(goal, conjecture, p(f(a))).")
    grounding = Grounding(problem, embedding_size=1)
    predicate = tf.keras.Sequential([tf.keras.layers.Dense(1)])
    grounding.predicates["p"] = ltn.Predicate.FromLogits(
        predicate, activation_function="sigmoid", with_class_indexing=False
    )
    function = tf.keras.Sequential([tf.keras.Sequential([tf.keras.layers.Dense(1)])])
    grounding.functions["f"] = ltn.Function(function)

    grounding.train(1)

    assert float(grounding.evaluate("s").tensor) > 0.5
    assert not predicate.built and not function.built


@pytest.mark.parametrize("trainable", [True, False])
def test_unbuilt_sequential_preserves_its_built_child_weights(trainable: bool) -> None:
    grounding, _, _ = lazy_grounding("predicate")
    layer = tf.keras.layers.Dense(1)
    layer.build((None, 1))
    model = tf.keras.Sequential([layer])
    model.trainable = trainable
    grounding.predicates["p"] = ltn.Predicate.FromLogits(
        model, activation_function="sigmoid", with_class_indexing=False
    )
    assert not model.built

    parameters = {id(parameter) for parameter in grounding.trainable_variables}

    assert all((id(weight) in parameters) == trainable for weight in layer.weights)


class _LazyBlock(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__(name="lazy_block")
        self.offset = self.add_weight(name="offset", shape=(), initializer="zeros")
        self.network = tf.keras.Sequential([tf.keras.layers.Dense(1)])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.network(inputs) + self.offset


class _NestedLazyModel(tf.keras.Model):
    def __init__(self, container_layer: bool = True) -> None:
        super().__init__(name="nested_lazy_model")
        self.offset = self.add_weight(name="offset", shape=(), initializer="zeros")
        self.ready = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")
        self.ready.build((None, 1))
        self.shared = self.ready
        self.lazy = (
            _LazyBlock() if container_layer else tf.keras.Sequential([tf.keras.layers.Dense(1)])
        )
        self.blocks = {"ready": [self.ready], "lazy": [self.lazy]}

    def build(self, input_shape: tf.TensorShape) -> None:
        # The ready branch is built in __init__; the other branch remains unused.
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # The unused branch stays lazy even after the containing model has been called.
        return self.ready(inputs) + self.offset


class TestNestedLazyModels(unittest.TestCase):
    def test_unused_subclasses_do_not_block_training_or_refutation(self) -> None:
        for container_layer in (False, True):
            with self.subTest(container_layer=container_layer):
                problem = tptp.parse("fof(fact,axiom,s). fof(goal,conjecture,p(a)).")
                grounding = Grounding(problem, embedding_size=1)
                model = _NestedLazyModel(container_layer)
                predicate = ltn.Predicate.FromLogits(
                    model, activation_function="sigmoid", with_class_indexing=False
                )
                grounding.predicates["p"] = predicate
                grounding.predicates["alias"] = predicate
                known = [model.offset, *model.ready.weights]
                if container_layer:
                    known.append(model.lazy.offset)

                parameters = grounding.trainable_variables
                for weight in known:
                    self.assertEqual(sum(parameter is weight for parameter in parameters), 1)
                grounding.train(1)
                self.assertGreater(float(grounding.propositions["s"].tensor), 0.5)
                result = grounding.refute("$true", steps=1)
                self.assertTrue(np.isfinite(result).all())
                self.assertFalse(model.built)
                self.assertFalse(model.lazy.built)
                for weight in known:
                    np.testing.assert_array_equal(weight.numpy(), np.zeros(weight.shape))

    def test_nested_parameters_preserve_parent_and_child_trainability(self) -> None:
        for frozen in (None, "parent", "ready", "lazy"):
            with self.subTest(frozen=frozen):
                grounding, _, _ = lazy_grounding("predicate")
                model = _NestedLazyModel()
                if frozen == "parent":
                    model.trainable = False
                elif frozen is not None:
                    getattr(model, frozen).trainable = False
                grounding.predicates["p"] = ltn.Predicate.FromLogits(
                    model, activation_function="sigmoid", with_class_indexing=False
                )
                parameters = {id(parameter) for parameter in grounding.trainable_variables}
                self.assertEqual(id(model.offset) in parameters, frozen != "parent")
                self.assertEqual(
                    id(model.lazy.offset) in parameters, frozen not in ("parent", "lazy")
                )
                for weight in model.ready.weights:
                    self.assertEqual(id(weight) in parameters, frozen not in ("parent", "ready"))

    def test_built_subclass_can_train_with_an_unused_lazy_branch(self) -> None:
        grounding, _, formula = lazy_grounding("predicate")
        model = _NestedLazyModel()
        grounding.predicates["p"] = ltn.Predicate.FromLogits(
            model, activation_function="sigmoid", with_class_indexing=False
        )

        grounding.train(1, optimizer=tf.keras.optimizers.SGD(0.1))

        self.assertTrue(model.built)
        self.assertFalse(model.lazy.built)
        self.assertGreater(float(model.offset), 0.0)
        self.assertGreater(float(grounding.evaluate(formula).tensor), 0.5)
        for weight in model.ready.weights:
            self.assertTrue(np.all(weight.numpy() > 0.0))
        self.assertEqual(float(model.lazy.offset), 0.0)

    def test_legacy_keras_preserves_tracked_module_weights(self) -> None:
        grounding, _, _ = lazy_grounding("predicate")
        model = _NestedLazyModel()
        if not hasattr(model, "_trainable_weights"):
            self.skipTest("Keras 3 does not track plain tf.Modules as model weights")
        model.extra = tf.Module()
        model.extra.weight = tf.Variable(0.0, trainable=True)
        grounding.predicates["p"] = ltn.Predicate.FromLogits(
            model, activation_function="sigmoid", with_class_indexing=False
        )
        own_weights = list(model._trainable_weights)

        first = grounding.trainable_variables
        second = grounding.trainable_variables

        self.assertIn(id(model.extra.weight), {id(parameter) for parameter in first})
        self.assertEqual([id(weight) for weight in first], [id(weight) for weight in second])
        self.assertEqual(
            [id(weight) for weight in model._trainable_weights],
            [id(weight) for weight in own_weights],
        )

    def test_unrelated_parameter_errors_are_not_suppressed(self) -> None:
        class InvalidModel(tf.keras.Model):
            @property
            def trainable_variables(self) -> list[tf.Variable]:
                raise ValueError("Invalid custom model state")

        grounding, _, _ = lazy_grounding("predicate")
        grounding.predicates["p"] = ltn.Predicate(InvalidModel())
        with self.assertRaisesRegex(ValueError, "Invalid custom model state"):
            _ = grounding.trainable_variables
