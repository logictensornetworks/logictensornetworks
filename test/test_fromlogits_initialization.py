"""FromLogits must build its child model correctly, without Keras build warnings."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf

import ltn


MODES = [('sigmoid', False), ('sigmoid', True), ('softmax', True)]


class StatefulLogits(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units, kernel_initializer='ones', bias_initializer='zeros')
        self.calls = self.add_weight(name='calls', shape=(), initializer='zeros', trainable=False)
        self.supports_masking = True

    def call(self, inputs, scale=1., offset=0., training=None, mask=None):
        self.calls.assign_add(1.)
        values = inputs[0]
        if isinstance(mask, (list, tuple)):
            mask = mask[0]
        if mask is not None:
            values = tf.where(mask, values, tf.zeros_like(values))
        return self.dense(values) * scale + offset


class TrainingLogits(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.norm = tf.keras.layers.BatchNormalization(momentum=.5)
        self.dropout = tf.keras.layers.Dropout(.5)
        kernel = np.arange(1, 4 * units + 1, dtype=np.float32).reshape(4, units) / 10
        self.dense = tf.keras.layers.Dense(units, kernel_initializer=tf.keras.initializers.Constant(kernel))

    def call(self, inputs, training=None):
        values = self.norm(inputs[0], training=training)
        return self.dense(self.dropout(values, training=training))


class ForwardingModel(tf.keras.Model):
    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def call(self, inputs, training=None):
        # The child must inherit the training context even without an explicit kwarg.
        return self.wrapped(inputs)


class TestFromLogitsInitialization(unittest.TestCase):
    def setUp(self):
        # A fallback that only marks the wrapper built must fail these tests.
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        self.addCleanup(warning_context.__exit__, None, None, None)
        warnings.simplefilter('error', UserWarning)

    def test_first_call_builds_once_with_arguments_and_gradients(self):
        for activation, indexing in MODES:
            for graph in [False, True]:
                for prebuilt in [False, True]:
                    with self.subTest(activation=activation, indexing=indexing, graph=graph, prebuilt=prebuilt):
                        model = StatefulLogits(2 if indexing else 1)
                        if prebuilt:
                            model([tf.ones([2, 2])])
                            model.dense.kernel.assign(tf.ones_like(model.dense.kernel) * .3)
                            weights_before = [id(w) for w in model.weights]
                            model.calls.assign(0.)
                        predicate = ltn.Predicate.FromLogits(model, activation, with_class_indexing=indexing)

                        def evaluate(values):
                            x = ltn.Variable('x', values)
                            inputs = [x, ltn.Variable('class', [0., 1.])] if indexing else [x]
                            with tf.GradientTape() as tape:
                                out = predicate(inputs, tf.constant(2.), offset=tf.constant(-.7)).tensor
                                loss = tf.reduce_sum(tf.square(out))
                            return out, tape.gradient(loss, predicate.trainable_variables)

                        runner = tf.function(evaluate, input_signature=[tf.TensorSpec([None, 2], tf.float32)]) if graph else evaluate
                        values = tf.constant([[.1, .2], [.3, .4]])
                        out, gradients = runner(values)
                        self.assertTrue(predicate.model.built)
                        self.assertTrue(model.built)
                        self.assertTrue(model.dense.built)
                        self.assertEqual(len(model.trainable_variables), 2)
                        self.assertEqual(float(model.calls), 1.)
                        self.assertTrue(all(g is not None and np.all(np.isfinite(g)) for g in gradients))
                        logits = (tf.matmul(values, model.dense.kernel) + model.dense.bias) * 2. - .7
                        expected = tf.nn.softmax(logits) if activation == 'softmax' else tf.sigmoid(logits)
                        np.testing.assert_allclose(out, expected if indexing else expected[:, 0], atol=1e-6)
                        if prebuilt:
                            self.assertEqual([id(w) for w in model.weights], weights_before)
                            np.testing.assert_allclose(model.dense.kernel, .3)
                        weights_after = [id(w) for w in model.weights]
                        runner(tf.ones([3, 2]))
                        self.assertEqual(float(model.calls), 2.)
                        self.assertEqual([id(w) for w in model.weights], weights_after)
                        self.assertEqual([id(w) for w in predicate.trainable_variables],
                                         [id(w) for w in model.trainable_variables])

    def test_training_updates_batchnorm_once_and_inference_preserves_it(self):
        values = tf.reshape(tf.range(128, dtype=tf.float32), [32, 4]) / 100
        for activation, indexing in MODES:
            for graph in [False, True]:
                with self.subTest(activation=activation, indexing=indexing, graph=graph):
                    model = TrainingLogits(2 if indexing else 1)
                    predicate = ltn.Predicate.FromLogits(model, activation, with_class_indexing=indexing)
                    inputs = [ltn.Variable('x', values)]
                    if indexing:
                        inputs.append(ltn.Constant(0., trainable=False))

                    def evaluate(training):
                        return predicate(inputs, training=training).tensor

                    runner = tf.function(evaluate) if graph else evaluate
                    training_out = runner(True)
                    np.testing.assert_allclose(model.norm.moving_mean, .5 * tf.reduce_mean(values, axis=0), atol=1e-6)
                    moving_mean = model.norm.moving_mean.numpy().copy()
                    moving_variance = model.norm.moving_variance.numpy().copy()
                    inference_out = runner(False)
                    np.testing.assert_array_equal(model.norm.moving_mean, moving_mean)
                    np.testing.assert_array_equal(model.norm.moving_variance, moving_variance)
                    self.assertFalse(np.allclose(training_out, inference_out))
                    np.testing.assert_allclose(runner(False), inference_out, atol=1e-6)

    def test_direct_model_call_forwards_explicit_mask(self):
        values = tf.constant([[.1, .2], [.3, .4]])
        mask = tf.constant([[True, False], [False, True]])
        for activation, indexing in MODES:
            with self.subTest(activation=activation, indexing=indexing):
                model = StatefulLogits(2 if indexing else 1)
                predicate = ltn.Predicate.FromLogits(model, activation, with_class_indexing=indexing)
                inputs = [values, tf.constant([[0.], [1.]])] if indexing else [values]
                out = predicate.model(inputs, mask=mask)
                logits = tf.matmul(tf.where(mask, values, 0.), model.dense.kernel)
                expected = tf.nn.softmax(logits) if activation == 'softmax' else tf.sigmoid(logits)
                if indexing:
                    expected = tf.gather(expected, [0, 1], batch_dims=1)
                np.testing.assert_allclose(out, expected, atol=1e-6)
                self.assertEqual(float(model.calls), 1.)

    def test_training_context_is_inherited_from_parent(self):
        values = tf.reshape(tf.range(32, dtype=tf.float32), [8, 4]) / 10
        for activation, indexing in MODES:
            with self.subTest(activation=activation, indexing=indexing):
                model = TrainingLogits(2 if indexing else 1)
                predicate = ltn.Predicate.FromLogits(model, activation, with_class_indexing=indexing)
                parent = ForwardingModel(predicate.model)
                inputs = [values, tf.zeros([8, 1])] if indexing else [values]
                parent(inputs, training=True)
                np.testing.assert_allclose(model.norm.moving_mean, .5 * tf.reduce_mean(values, axis=0), atol=1e-6)
                before = model.norm.moving_mean.numpy().copy()
                parent(inputs, training=False)
                np.testing.assert_array_equal(model.norm.moving_mean, before)

    def test_functional_model_replays_custom_arguments(self):
        for activation, indexing in MODES:
            with self.subTest(activation=activation, indexing=indexing):
                model = StatefulLogits(2 if indexing else 1)
                predicate = ltn.Predicate.FromLogits(model, activation, with_class_indexing=indexing)
                x, scale = tf.keras.Input((2,)), tf.keras.Input((1,))
                raw_inputs = [x]
                model_inputs = [x, scale]
                values = tf.constant([[.1, .2], [.3, .4]])
                supplied = [values, tf.constant([[2.], [3.]])]
                if indexing:
                    classes = tf.keras.Input((1,))
                    raw_inputs.append(classes)
                    model_inputs.append(classes)
                    supplied.append(tf.constant([[0.], [1.]]))
                outputs = predicate.model(raw_inputs, scale, offset=.5)
                functional = tf.keras.Model(model_inputs, outputs)
                out = functional(supplied)
                logits = tf.matmul(values, model.dense.kernel) * supplied[1] + .5
                expected = tf.nn.softmax(logits) if activation == 'softmax' else tf.sigmoid(logits)
                if indexing:
                    expected = tf.gather(expected, [0, 1], batch_dims=1)
                np.testing.assert_allclose(out, expected, atol=1e-6)
                self.assertEqual(float(model.calls), 1.)

    def test_weights_round_trip(self):
        values = tf.constant([[.1, .2], [.3, .4]])
        for activation, indexing in MODES:
            with self.subTest(activation=activation, indexing=indexing):
                model = StatefulLogits(2 if indexing else 1)
                predicate = ltn.Predicate.FromLogits(model, activation, with_class_indexing=indexing)
                inputs = [values, tf.constant([[0.], [1.]])] if indexing else [values]
                predicate.model(inputs)
                model.dense.kernel.assign(tf.ones_like(model.dense.kernel) * .7)
                expected = predicate.model(inputs)
                with tempfile.TemporaryDirectory() as directory:
                    path = str(Path(directory) / 'logits.weights.h5')
                    predicate.model.save_weights(path)
                    restored = ltn.Predicate.FromLogits(StatefulLogits(2 if indexing else 1), activation,
                                                        with_class_indexing=indexing)
                    restored.model(inputs)
                    restored.model.load_weights(path)
                    np.testing.assert_allclose(restored.model(inputs), expected, atol=1e-6)
