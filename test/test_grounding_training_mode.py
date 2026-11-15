"""Training-sensitive custom groundings use the requested Keras mode."""

import numpy as np
import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


def fixed_grounding(source: str, **options) -> Grounding:
    """Give the two domain individuals known scalar features."""
    grounding = Grounding(tptp.parse(source), embedding_size=1, **options)
    grounding.constants["a"] = ltn.Constant([1.0], trainable=False)
    grounding.constants["b"] = ltn.Constant([3.0], trainable=False)
    return grounding


def batch_norm_model() -> tuple[tf.keras.Model, tf.keras.layers.BatchNormalization]:
    """Build eagerly so these tests isolate training mode from lazy initialization."""
    norm = tf.keras.layers.BatchNormalization(momentum=0.5)
    model = tf.keras.Sequential([tf.keras.Input(shape=(1,)), norm])
    return model, norm


def test_train_updates_batch_normalization_and_inference_does_not() -> None:
    grounding = fixed_grounding("fof(fact,axiom, ![X]:p(X)). fof(a,axiom,a=a). fof(b,axiom,b=b).")
    model, norm = batch_norm_model()
    predicate = tf.keras.Sequential([model, tf.keras.layers.Activation("sigmoid")])
    predicate(tf.zeros((1, 1)))
    grounding.predicates["p"] = ltn.Predicate(predicate)

    grounding.train(1, tf.keras.optimizers.SGD(0.0), reuse_graph=True)
    np.testing.assert_allclose(norm.moving_mean.numpy(), [1.0])
    grounding.train(1, reuse_graph=True)
    np.testing.assert_allclose(norm.moving_mean.numpy(), [1.5])

    before = norm.moving_mean.numpy().copy()
    grounding.sat()
    grounding.evaluate("![X]:p(X)")
    grounding.train(0)
    grounding.refute("p(a)", steps=2, optimizer=tf.keras.optimizers.SGD(0.0))
    np.testing.assert_array_equal(norm.moving_mean.numpy(), before)


def test_train_enables_dropout_while_evaluation_stays_deterministic() -> None:
    grounding = fixed_grounding("fof(fact,axiom,p(a)).")
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Activation("sigmoid"),
        ]
    )
    grounding.predicates["p"] = ltn.Predicate(model)
    inference = float(grounding.sat())
    trained = grounding.train(1, tf.keras.optimizers.SGD(0.0))
    # With one nonzero feature the dropout output is either zero or twice the input.
    assert abs(trained - inference) > 0.1
    assert float(grounding.sat()) == pytest.approx(inference)
    assert float(grounding.evaluate("p(a)").tensor) == pytest.approx(tf.math.sigmoid(1.0))


@pytest.mark.parametrize("herbrand_depth", [0, 1])
def test_training_mode_reaches_functions_and_herbrand_terms(herbrand_depth: int) -> None:
    # f is used directly at depth 0, and only to build the domain at depth 1.
    formula = "p(f(X))" if herbrand_depth == 0 else "p(X)"
    grounding = fixed_grounding(
        f"fof(fact,axiom, ![X]:{formula}). "
        "fof(a,axiom,a=a). fof(b,axiom,b=b). fof(f,conjecture,p(f(a))).",
        herbrand_depth=herbrand_depth,
    )
    model, norm = batch_norm_model()
    grounding.functions["f"] = ltn.Function(model)
    grounding.predicates["p"] = ltn.Predicate.Lambda(lambda x: tf.math.sigmoid(x[:, 0]))

    grounding.train(1, tf.keras.optimizers.SGD(0.0))
    assert norm.moving_mean.numpy()[0] > 0.0
    before = norm.moving_mean.numpy().copy()
    grounding.individuals()
    grounding.evaluate("![X]:p(X)")
    np.testing.assert_array_equal(norm.moving_mean.numpy(), before)


def test_training_mode_reaches_implicit_distinct_value_equality() -> None:
    grounding = Grounding(tptp.parse("fof(a,axiom,p(1)&p(2))."), embedding_size=1)
    grounding.constants["1"] = ltn.Constant([1.0], trainable=False)
    grounding.constants["2"] = ltn.Constant([3.0], trainable=False)
    norm = tf.keras.layers.BatchNormalization(momentum=0.5)
    left = tf.keras.Input(shape=(1,))
    right = tf.keras.Input(shape=(1,))
    difference = tf.keras.layers.Subtract()([norm(left), right])
    equal = tf.keras.layers.Activation("sigmoid")(difference)
    grounding.predicates["="] = ltn.Predicate(tf.keras.Model([left, right], equal))

    grounding.train(1, tf.keras.optimizers.SGD(0.0))
    np.testing.assert_allclose(norm.moving_mean.numpy(), [1.0])
    before = norm.moving_mean.numpy().copy()
    grounding.sat()
    np.testing.assert_array_equal(norm.moving_mean.numpy(), before)
