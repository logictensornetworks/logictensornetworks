"""Custom models must be built by the first forward pass before collecting their weights."""

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
