import subprocess
import sys
import unittest

import numpy as np
import tensorflow as tf

import ltn
from ltn import core, fuzzy_ops


def array_allclose(actual, desired):
    try:
        return np.allclose(actual, desired) and (actual.shape == desired.shape)
    except AttributeError:
        actual, desired = np.array(actual), np.array(desired)
        return np.allclose(actual, desired) and (actual.shape == desired.shape)


class ListInputLogits(tf.keras.Model):
    """Logits model called with a list of inputs, as required by `Predicate.FromLogits`."""

    def __init__(self, n_classes, dropout=0.0):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(n_classes)

    def call(self, inputs, training=False):
        return self.dense(self.dropout(inputs[0], training=training))


class TestConstant(unittest.TestCase):
    def setUp(self):
        self.c_val_1 = [2.1, 3.0]
        self.c_val_2 = [[4.2, 3, 2.5], [4, -1.3, 1.8]]
        self.c_val_3 = [4.1, 0.5]
        self.c_vals = [self.c_val_1, self.c_val_2, self.c_val_3]
        self.c_val_4 = 1.0

    def test_init_from_python(self):
        for c_val in self.c_vals:
            c = core.Constant(c_val, trainable=False)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Tensor))
            self.assertEqual(c.free_vars, [])

    def test_init_from_np(self):
        for c_val in self.c_vals:
            c_val = np.array(c_val)
            c = core.Constant(c_val, trainable=False)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Tensor))
            self.assertEqual(c.free_vars, [])

    def test_init_from_tf(self):
        for c_val in self.c_vals:
            c_val = tf.constant(c_val)
            c = core.Constant(c_val, trainable=False)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Tensor))
            self.assertEqual(c.free_vars, [])

    def test_cast(self):
        for c_val in [
            [1, 2],
            np.array([1, 2]),
            tf.constant([1, 2]),
            tf.constant([1.0, 2.0], dtype=tf.float64),
        ]:
            for trainable in [False, True]:
                with self.subTest(c_val=c_val, trainable=trainable):
                    c = core.Constant(c_val, trainable=trainable)
                    self.assertEqual(c.tensor.dtype, tf.float32)
                    self.assertTrue(array_allclose(c.tensor.numpy(), np.array([1.0, 2.0])))

    def test_expand_dims(self):
        # rationale for expand dims:
        #  P([x,y]), if x.shape [free_dim,1] and y.shape [free_dim], the broadcast is broken
        for trainable in [False, True]:
            with self.subTest(trainable=trainable):
                c = core.Constant(self.c_val_4, trainable=trainable)
                self.assertEqual(c.tensor.shape, [1])
                self.assertTrue(array_allclose(c.tensor.numpy(), np.array([self.c_val_4])))
        # indexing a variable returns a tensor: a trainable scalar must still be a variable after
        # the expansion
        c = core.Constant(self.c_val_4, trainable=True)
        self.assertTrue(isinstance(c.tensor, tf.Variable))
        with tf.GradientTape() as tape:
            res = core.Predicate.Lambda(lambda x: tf.reduce_sum(x, axis=1))(c)
        self.assertTrue(tape.gradient(res.tensor, c.tensor) is not None)

    def test_trainable(self):
        for c_val in self.c_vals:
            c = core.Constant(c_val, trainable=True)
            self.assertTrue(array_allclose(c.tensor.numpy(), c_val))
            self.assertTrue(isinstance(c.tensor, tf.Variable))
            self.assertTrue(c.tensor.trainable)
            self.assertEqual(c.free_vars, [])


class TestVariable(unittest.TestCase):
    def setUp(self):
        self.x_val_1 = np.random.rand(10, 2)  # 10 individuals in R^2
        self.x_val_2 = np.random.rand(5, 2)  # 5 individuals in R^2
        self.x_val_3 = np.random.rand(7, 2, 2)  # 7 individuals in R^(2x2)
        self.x_vals = [self.x_val_1, self.x_val_2, self.x_val_3]
        self.x_val_4 = np.random.rand(10)  # 10 inviduals in R

    def test_init(self):
        for x_val in self.x_vals:
            label = "x"
            x = core.Variable("x", x_val)
            self.assertTrue(array_allclose(x.tensor.numpy(), x_val))
            self.assertTrue(isinstance(x.tensor, tf.Tensor))
            self.assertEqual(x.free_vars, [label])
            self.assertEqual(x.label, label)

    def test_from_trainable_constants(self):
        c1 = core.Constant([2.1, 3], trainable=True)
        c2 = core.Constant([4.5, 0.8], trainable=True)
        with tf.GradientTape() as tape:
            x = core.Variable.from_constants("x", [c1, c2], tape=tape)
        self.assertTrue(tape.gradient(x.tensor, c1.tensor) is not None)

        tape = tf.GradientTape()
        with self.assertRaises(ValueError):
            x = core.Variable.from_constants("x", [c1, c2], tape=tape)

    def test_from_non_trainable_constants(self):
        c1 = core.Constant([2.1, 3], trainable=False)
        c2 = core.Constant([4.5, 0.8], trainable=False)
        with tf.GradientTape() as tape:
            x = core.Variable.from_constants("x", [c1, c2], tape=None)
        self.assertTrue(tape.gradient(x.tensor, c1.tensor) is None)

    def test_from_constants_without_tape_warns(self):
        c1 = core.Constant([2.1, 3], trainable=True)
        with self.assertWarns(UserWarning):
            core.Variable.from_constants("x", [c1])

    def test_cast(self):
        for x_val in [
            [[1], [2]],
            np.array([[1], [2]]),
            tf.constant([[1], [2]]),
            tf.constant([[1.0], [2.0]], dtype=tf.float64),
            [[True], [False]],
        ]:
            with self.subTest(x_val=x_val):
                x = core.Variable("x", x_val)
                self.assertEqual(x.tensor.dtype, tf.float32)
                self.assertTrue(array_allclose(x.tensor.numpy(), np.array(x_val, dtype=np.float32)))

    def test_expand_dims(self):
        # individuals given as scalars get a feature dim, like individuals with one feature
        x = core.Variable("x", self.x_val_4)
        self.assertEqual(x.tensor.shape, [10, 1])
        self.assertTrue(array_allclose(x.tensor.numpy()[:, 0], self.x_val_4))
        y = core.Variable("y", np.random.rand(5, 1))
        add = core.Function.Lambda(lambda args: args[0] + args[1])
        self.assertEqual(add([x, y]).tensor.shape, [10, 5, 1])


class TestPredicate(unittest.TestCase):
    def setUp(self):
        self.c1 = core.Constant([2.1, 3], trainable=False)
        self.c2 = core.Constant([4.5, 0.8], trainable=False)
        self.n_x = 10
        self.x = core.Variable("x", np.random.normal(0.0, 1.0, (self.n_x, 2)))
        self.n_y = 5
        self.y = core.Variable("y", np.random.normal(0.0, 4.0, (self.n_y, 2)))

    def test_from_tf_model_1input(self):
        class ModelP(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

            def call(self, x):
                return self.dense1(x)

        P = core.Predicate(ModelP())
        # Produces correct result on constant
        self.assertEqual(P(self.c1).tensor, P.model(tf.expand_dims(self.c1.tensor, axis=0)))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(P(self.x).tensor, tf.squeeze(P.model(self.x.tensor))))
        self.assertEqual(P(self.x).free_vars, ["x"])
        self.assertEqual(P(self.x)._get_dim_of_free_var("x"), self.n_x)

    def test_from_lambda_1input(self):
        mu = tf.constant([2.0, 3.0])
        P = core.Predicate.Lambda(lambda x: tf.exp(-tf.norm(x - mu, axis=1)))
        # Produces correct result on constant
        self.assertEqual(P(self.c1).tensor, P.model(tf.expand_dims(self.c1.tensor, axis=0)))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(P(self.x).tensor, P.model(self.x.tensor)))
        self.assertEqual(P(self.x).free_vars, ["x"])
        self.assertEqual(P(self.x)._get_dim_of_free_var("x"), self.n_x)

    def test_from_default_MLP_1input(self):
        P = core.Predicate.MLP(input_shapes=[2])
        # Produces correct result on constant
        self.assertEqual(P(self.c1).tensor, P.model(tf.expand_dims(self.c1.tensor, axis=0)))
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(P(self.x).tensor, tf.squeeze(P.model(self.x.tensor))))
        self.assertEqual(P(self.x).free_vars, ["x"])
        self.assertEqual(P(self.x)._get_dim_of_free_var("x"), self.n_x)

    def test_from_tf_model_2inputs(self):
        class ModelP(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

            def call(self, inputs):
                x = tf.concat([inputs[0], inputs[1]], axis=1)
                return self.dense1(x)

        P = core.Predicate(ModelP())
        # Produces correct result on constant
        self.assertEqual(
            P([self.c1, self.c2]).tensor,
            P.model(
                [tf.expand_dims(self.c1.tensor, axis=0), tf.expand_dims(self.c2.tensor, axis=0)]
            ),
        )
        # Produces correct outputs with variable
        self.assertTrue(
            array_allclose(
                P([self.x, self.y]).take("x", 0).take("y", 0).tensor,
                tf.squeeze(P.model([tf.gather(self.x.tensor, [0]), tf.gather(self.y.tensor, [0])])),
            )
        )
        self.assertEqual(sorted(P([self.x, self.y]).free_vars), ["x", "y"])
        self.assertEqual(P([self.x, self.y])._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(P([self.x, self.y])._get_dim_of_free_var("y"), self.n_y)

    def test_from_lambda_2inputs(self):
        P = core.Predicate.Lambda(lambda args: tf.exp(-tf.norm(args[0] - args[1], axis=1)))
        # Produces correct result on constant
        self.assertEqual(
            P([self.c1, self.c2]).tensor,
            P.model(
                [tf.expand_dims(self.c1.tensor, axis=0), tf.expand_dims(self.c2.tensor, axis=0)]
            ),
        )
        # Produces correct outputs with variable
        self.assertTrue(
            array_allclose(
                P([self.x, self.y]).take("x", 0).take("y", 0).tensor,
                tf.squeeze(P.model([tf.gather(self.x.tensor, [0]), tf.gather(self.y.tensor, [0])])),
            )
        )
        self.assertEqual(sorted(P([self.x, self.y]).free_vars), ["x", "y"])
        self.assertEqual(P([self.x, self.y])._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(P([self.x, self.y])._get_dim_of_free_var("y"), self.n_y)

    def test_from_default_MLP_2inputs(self):
        P = core.Predicate.MLP(input_shapes=[2, 2])
        # Produces correct result on constant
        self.assertTrue(
            array_allclose(
                tf.squeeze(P([self.c1, self.c2]).tensor),
                tf.squeeze(
                    P.model(
                        [
                            tf.expand_dims(self.c1.tensor, axis=0),
                            tf.expand_dims(self.c2.tensor, axis=0),
                        ]
                    )
                ),
            )
        )
        # Produces correct outputs with variable
        self.assertTrue(
            array_allclose(
                tf.squeeze(P([self.x, self.y]).take("x", 0).take("y", 0).tensor),
                tf.squeeze(P.model([tf.gather(self.x.tensor, [0]), tf.gather(self.y.tensor, [0])])),
            )
        )
        self.assertEqual(sorted(P([self.x, self.y]).free_vars), ["x", "y"])
        self.assertEqual(P([self.x, self.y])._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(P([self.x, self.y])._get_dim_of_free_var("y"), self.n_y)

    def test_passes_kwargs_to_model(self):
        """e.g. `training=True` reaches the dropout layers, also through `FromLogits`"""
        x = core.Variable("x", np.random.normal(0.0, 1.0, (64, 8)))
        label = core.Constant(0, trainable=False)
        dropout_model = tf.keras.Sequential(
            [tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)]
        )
        predicates_and_inputs = [
            (core.Predicate(dropout_model), x),
            (
                core.Predicate.FromLogits(
                    ListInputLogits(3, dropout=0.5), activation_function="softmax"
                ),
                [x, label],
            ),
            (
                core.Predicate.FromLogits(
                    ListInputLogits(3, dropout=0.5), activation_function="sigmoid"
                ),
                [x, label],
            ),
        ]
        for P, inputs in predicates_and_inputs:
            with self.subTest(model=P.model.__class__.__name__):
                inference = P(inputs, training=False).tensor
                self.assertTrue(array_allclose(P(inputs).tensor, inference))
                self.assertFalse(np.allclose(P(inputs, training=True).tensor, inference))

    def test_trainable_variables_warns_when_empty(self):
        P = core.Predicate.Lambda(lambda x: tf.reduce_sum(x, axis=1))
        with self.assertWarns(UserWarning):
            self.assertEqual(P.trainable_variables, [])


class TestFunction(unittest.TestCase):
    def setUp(self):
        self.c1 = core.Constant([2.1, 3], trainable=False)
        self.c2 = core.Constant([4.5, 0.8], trainable=False)
        self.n_x = 10
        self.x = core.Variable("x", np.random.normal(0.0, 1.0, (self.n_x, 2)))
        self.n_y = 5
        self.y = core.Variable("y", np.random.normal(0.0, 4.0, (self.n_y, 2)))

    def test_from_tf_model(self):
        class ModelF(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = tf.keras.layers.Dense(5)

            def call(self, x):
                return self.dense1(x)

        f = core.Function(ModelF())
        # Produces correct result on constant
        self.assertTrue(
            array_allclose(f(self.c1).tensor, f.model(tf.expand_dims(self.c1.tensor, axis=0))[0])
        )
        # Produces correct outputs with variable
        self.assertTrue(array_allclose(f(self.x).tensor, f.model(self.x.tensor)))
        self.assertEqual(f(self.x).free_vars, ["x"])
        self.assertEqual(f(self.x)._get_dim_of_free_var("x"), self.n_x)

    def test_from_lambda(self):
        f = core.Function.Lambda(lambda args: args[0] - args[1])
        # Produces correct result on constant
        self.assertTrue(
            array_allclose(
                f([self.c1, self.c2]).tensor,
                f.model(
                    [tf.expand_dims(self.c1.tensor, axis=0), tf.expand_dims(self.c2.tensor, axis=0)]
                )[0],
            )
        )
        # Produces correct outputs with variable
        self.assertTrue(
            array_allclose(
                f([self.x, self.y]).take("x", 0).take("y", 0).tensor,
                f.model([tf.gather(self.x.tensor, [0]), tf.gather(self.y.tensor, [0])])[0],
            )
        )
        self.assertEqual(sorted(f([self.x, self.y]).free_vars), ["x", "y"])
        self.assertEqual(f([self.x, self.y])._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(f([self.x, self.y])._get_dim_of_free_var("y"), self.n_y)

    def test_from_default_MLP(self):
        f = core.Function.MLP(input_shapes=[2, 2], output_shape=[2])
        # Produces correct result on constant
        self.assertTrue(
            array_allclose(
                f([self.c1, self.c2]).tensor,
                f.model(
                    [tf.expand_dims(self.c1.tensor, axis=0), tf.expand_dims(self.c2.tensor, axis=0)]
                )[0],
            )
        )
        # Produces correct outputs with variable
        self.assertTrue(
            array_allclose(
                f([self.x, self.y]).take("x", 0).take("y", 0).tensor,
                f.model([tf.gather(self.x.tensor, [0]), tf.gather(self.y.tensor, [0])])[0],
            )
        )
        self.assertEqual(sorted(f([self.x, self.y]).free_vars), ["x", "y"])
        self.assertEqual(f([self.x, self.y])._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(f([self.x, self.y])._get_dim_of_free_var("y"), self.n_y)


class TestProposition(unittest.TestCase):
    def setUp(self):
        self.And = core.Wrapper_Connective(fuzzy_ops.And_Prod())

    def test_trainable(self):
        a = core.Proposition(0.0, trainable=True)
        b = core.Proposition(0.0, trainable=True)
        with tf.GradientTape() as tape:
            res = self.And(a, b)
        self.assertTrue(tape.gradient(res.tensor, a.tensor) is not None)

    def test_non_trainable(self):
        a = core.Proposition(0.0, trainable=False)
        b = core.Proposition(0.0, trainable=False)
        with tf.GradientTape() as tape:
            res = self.And(a, b)
        self.assertTrue(tape.gradient(res.tensor, a.tensor) is None)

    def test_invalid_truth_values(self):
        for value in [-0.1, 1.5, float("nan"), "a", None]:
            with self.assertRaises(ValueError):
                core.Proposition(value, trainable=False)

    def test_invalid_truth_value_with_optimizations(self):
        """The range check must not rely on `assert`, which `python -O` strips."""
        code = (
            "import ltn\n"
            "try:\n"
            "    ltn.Proposition(1.5, trainable=False)\n"
            "except ValueError:\n"
            "    print('raised')\n"
        )
        result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
        self.assertIn("raised", result.stdout)


class TestBroadcast(unittest.TestCase):
    def setUp(self):
        self.var_settings = {
            "x1": {"n_individuals": 10, "shape_individual": [2]},
            "x2": {"n_individuals": 5, "shape_individual": [2]},
            "x3": {"n_individuals": 7, "shape_individual": [2, 2]},
        }
        self.xs = {}
        for label, v_s in self.var_settings.items():
            x = core.Variable(label, np.random.rand(v_s["n_individuals"], *v_s["shape_individual"]))
            self.xs[label] = x

    def test_free_dims_broadcast_variables(self):
        """Adds the correct dimensions for the free variables."""
        exprs = core.broadcast_exprs(list(self.xs.values()), in_place=False)
        for expr in exprs:
            self.assertEqual(sorted(expr.free_vars), sorted(self.var_settings.keys()))
            for label, v_s in self.var_settings.items():
                self.assertEqual(expr._get_dim_of_free_var(label).numpy(), v_s["n_individuals"])

    def test_broadcast_variables_intact(self):
        """Keeps the correct values"""
        exprs = core.broadcast_exprs(list(self.xs.values()), in_place=False)
        for expr, x in zip(exprs, list(self.xs.values()), strict=True):
            other_free_vars = [x_.label for x_ in list(self.xs.values()) if x_.label != x.label]
            for free_var in other_free_vars:
                expr = expr.take(free_var, 0)
            self.assertTrue(array_allclose(x.tensor, expr.tensor))

    def test_free_dims_wt_broadcast_constant(self):
        """Adds the correct dimensions if a constant is involved"""
        c1 = core.Constant([2.1, 3], trainable=False)
        exprs = core.broadcast_exprs(list(self.xs.values()) + [c1])
        for expr in exprs:
            self.assertEqual(sorted(expr.free_vars), sorted(self.var_settings.keys()))
            for label, v_s in self.var_settings.items():
                self.assertEqual(expr._get_dim_of_free_var(label).numpy(), v_s["n_individuals"])

    def test_broadcast_formulas(self):
        """Adds the correct dimensions between formulas"""
        p1 = core.Predicate.MLP(
            input_shapes=[
                self.var_settings["x1"]["shape_individual"],
                self.var_settings["x2"]["shape_individual"],
            ]
        )
        p2 = core.Predicate.MLP(input_shapes=[self.var_settings["x1"]["shape_individual"]])
        p3 = core.Predicate.MLP(input_shapes=[self.var_settings["x3"]["shape_individual"]])
        phi1 = p1([self.xs["x1"], self.xs["x2"]])
        phi2 = p2([self.xs["x1"]])
        phi3 = p3([self.xs["x3"]])
        exprs = core.broadcast_exprs([phi1, phi2, phi3])
        for expr in exprs:
            self.assertEqual(sorted(expr.free_vars), sorted(self.var_settings.keys()))
            for label, v_s in self.var_settings.items():
                self.assertEqual(expr._get_dim_of_free_var(label).numpy(), v_s["n_individuals"])

    def test_in_place(self):
        xs = list(self.xs.values())
        copies = core.broadcast_exprs(xs, in_place=False)
        for x, copy in zip(xs, copies, strict=True):
            self.assertIsNot(copy, x)
            self.assertEqual(x.free_vars, [x.label])
        exprs = core.broadcast_exprs(xs, in_place=True)
        for x, expr, copy in zip(xs, exprs, copies, strict=True):
            self.assertIs(expr, x)
            self.assertEqual(x.free_vars, copy.free_vars)
            self.assertTrue(array_allclose(x.tensor, copy.tensor))


class TestConnective(unittest.TestCase):
    def setUp(self):
        self.And = core.Wrapper_Connective(fuzzy_ops.And_Prod())
        self.Not = core.Wrapper_Connective(fuzzy_ops.Not_Std())
        self.c1 = core.Constant([2.1, 3], trainable=True)
        self.n_x = 10
        self.x = core.Variable("x", np.random.normal(0.0, 1.0, (self.n_x, 2)))
        self.n_y = 5
        self.y = core.Variable("y", np.random.normal(0.0, 4.0, (self.n_y, 2)))
        self.p1 = core.Predicate.MLP([2])
        self.p2 = core.Predicate.MLP([2, 2])
        self.a = core.Proposition(0.0, trainable=True)

    def test_unary_connective_variables(self):
        """Unary connective maintains variables"""
        res = self.Not(self.p1(self.x))
        self.assertEqual(res.free_vars, ["x"])
        self.assertEqual(res._get_dim_of_free_var("x"), self.n_x)

    def test_binary_connective_variables(self):
        """Binary connectives joins variables"""
        res = self.And(self.p1(self.x), self.p2([self.x, self.y]))
        self.assertEqual(sorted(res.free_vars), ["x", "y"])
        self.assertEqual(res._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(res._get_dim_of_free_var("y"), self.n_y)

    def test_binary_connective_wt_constant(self):
        """Binary connective doesn't break with constant"""
        res = self.And(self.p1(self.c1), self.p2([self.x, self.c1]))
        self.assertEqual(res.free_vars, ["x"])
        self.assertEqual(res._get_dim_of_free_var("x"), self.n_x)

    def test_binary_connective_wt_proposition(self):
        """Binary connective doesn't break with proposition"""
        res = self.And(self.a, self.p2([self.x, self.y]))
        self.assertEqual(sorted(res.free_vars), ["x", "y"])
        self.assertEqual(res._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(res._get_dim_of_free_var("y"), self.n_y)

    def test_gradients(self):
        """Tracks gradients"""
        with tf.GradientTape() as tape:
            res = self.And(self.a, self.p2([self.x, self.y]))
        self.assertTrue(tape.gradient(res.tensor, self.a.tensor) is not None)

    def test_values_correct(self):
        """Test if the value in the result is correct"""
        phi1 = self.p1(self.x)
        phi2 = self.p2([self.x, self.y])
        res = self.And(phi1, phi2)
        self.assertEqual(
            res.take("x", 0).take("y", 0).tensor,
            self.And.connective_op(phi1.take("x", 0).tensor, phi2.take("x", 0).take("y", 0).tensor),
        )

    def test_incompatible_shapes_raise_value_error(self):
        MatMul = core.Wrapper_Connective(lambda a, b: tf.matmul(a, b))
        with self.assertRaisesRegex(ValueError, "Could not connect formulas"):
            MatMul(self.p1(self.x), self.p1(self.y))

    def test_passes_kwargs(self):
        a = core.Proposition(0.0, trainable=False)
        b = core.Proposition(1.0, trainable=False)
        self.assertTrue(float(self.And(a, b).tensor) > 0.0)  # the stable product never reaches 0
        self.assertEqual(float(self.And(a, b, stable=False).tensor), 0.0)


class TestQuantifier(unittest.TestCase):
    def setUp(self):
        self.Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
        self.Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMean(p=5), semantics="exists")
        self.n_x = 10
        self.x = core.Variable("x", np.random.normal(0.0, 1.0, (self.n_x, 2)))
        self.n_y = 5
        self.y = core.Variable("y", np.random.normal(0.0, 4.0, (self.n_y, 2)))
        self.n_z = 5
        self.z = core.Variable("z", np.random.normal(0.0, 4.0, (self.n_z, 2)))
        self.p1 = core.Predicate.MLP([2, 2])
        self.p2 = core.Predicate.MLP([2, 2, 2])

    def test_aggreg_one_var(self):
        res = self.Forall(self.x, self.p1([self.x, self.y]))
        self.assertEqual(sorted(res.free_vars), ["y"])
        self.assertEqual(res._get_dim_of_free_var("y"), self.n_y)

        res = self.Forall(self.y, self.p1([self.x, self.y]))
        self.assertEqual(sorted(res.free_vars), ["x"])
        self.assertEqual(res._get_dim_of_free_var("x"), self.n_x)

    def test_aggreg_several_vars(self):
        res = self.Forall((self.x, self.y), self.p1([self.x, self.y]))
        self.assertEqual(sorted(res.free_vars), [])
        self.assertEqual(res.tensor.shape, [])

    def test_gradients(self):
        """Tracks gradients"""
        c_s = [core.Constant(np.random.rand(2), trainable=True) for _ in range(self.n_x)]
        with tf.GradientTape() as tape:
            x = core.Variable.from_constants("x", c_s, tape)
            phi = self.p1([x, self.y])
            res = self.Forall(x, phi)
        self.assertTrue(tape.gradient(res.tensor, c_s[0].tensor) is not None)

    def test_values_correct(self):
        phi = self.p2([self.x, self.y, self.z])
        self.assertTrue(
            array_allclose(
                self.Forall((self.y), phi).take("x", 0).take("z", 0).tensor,
                self.Forall.aggreg_op(phi.tensor, axis=phi._get_axis_of_free_var("y"))[0, 0],
            )
        )
        self.assertTrue(
            array_allclose(
                self.Forall((self.x, self.z), phi).take("y", 0).tensor,
                self.Forall.aggreg_op(
                    phi.tensor,
                    axis=[phi._get_axis_of_free_var("x"), phi._get_axis_of_free_var("z")],
                )[0],
            )
        )

    def test_passes_kwargs(self):
        """e.g. `Forall(x, phi, p=5)`, also with a mask"""
        x = core.Variable("x", np.random.rand(4, 1))
        wff = core.Formula(tf.constant([0.2, 0.4, 0.6, 1.0]), free_vars=["x"])
        mask = core.Formula(tf.constant([1.0, 1.0, 0.0, 1.0]), free_vars=["x"])
        Forall = core.Wrapper_Quantifier(
            fuzzy_ops.Aggreg_pMeanError(p=2, stable=False), semantics="forall"
        )
        # with p=1, the p-mean error is the mean
        self.assertTrue(array_allclose(Forall(x, wff, p=1).tensor, np.mean([0.2, 0.4, 0.6, 1.0])))
        self.assertTrue(
            array_allclose(Forall(x, wff, mask=mask, p=1).tensor, np.mean([0.2, 0.4, 1.0]))
        )


class TestTransposeFreeVars(unittest.TestCase):
    def setUp(self):
        self.n_x = 10
        self.x = core.Variable("x", np.random.normal(0.0, 1.0, (self.n_x, 2)))
        self.n_y = 5
        self.y = core.Variable("y", np.random.normal(0.0, 4.0, (self.n_y, 2)))
        self.n_z = 3
        self.z = core.Variable("z", np.random.normal(0.0, 4.0, (self.n_z, 2)))
        self.p = core.Predicate.MLP([2, 2, 2])

    def test_transpose(self):
        phi = self.p([self.x, self.y, self.z])
        var_order = ["x", "y", "z"]
        phi = core.transpose_free_vars(phi, var_order)
        self.assertEqual(phi.free_vars, var_order)
        self.assertEqual(phi._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(phi._get_axis_of_free_var("x"), var_order.index("x"))
        var_order = ["z", "x", "y"]
        phi = core.transpose_free_vars(phi, var_order)
        self.assertEqual(phi.free_vars, var_order)
        self.assertEqual(phi._get_dim_of_free_var("x"), self.n_x)
        self.assertEqual(phi._get_axis_of_free_var("x"), var_order.index("x"))
        self.assertEqual(phi._get_dim_of_free_var("z"), self.n_z)
        self.assertEqual(phi._get_axis_of_free_var("z"), var_order.index("z"))

    def test_conserves_values(self):
        phi = self.p([self.x, self.y, self.z])
        phi1 = core.transpose_free_vars(phi, ["x", "y", "z"])
        phi2 = core.transpose_free_vars(phi1, ["z", "x", "y"])
        phi3 = core.transpose_free_vars(phi2, ["x", "y", "z"])
        self.assertTrue(array_allclose(phi1.tensor, phi3.tensor))

    def test_in_place(self):
        phi = self.p([self.x, self.y, self.z])
        copy = core.transpose_free_vars(phi, ["z", "x", "y"])
        self.assertIsNot(copy, phi)
        self.assertEqual(phi.free_vars, ["x", "y", "z"])
        res = core.transpose_free_vars(phi, ["z", "x", "y"], in_place=True)
        self.assertIs(res, phi)
        self.assertEqual(phi.free_vars, ["z", "x", "y"])
        self.assertTrue(array_allclose(phi.tensor, copy.tensor))


class TestBroadcastToMask(unittest.TestCase):
    def setUp(self):
        self.n_x = 10
        self.x = core.Variable("x", np.random.normal(0.0, 1.0, (self.n_x, 1)))
        self.n_y = 10
        self.y = core.Variable("y", np.random.normal(0.0, 1.0, (self.n_y, 1)))
        self.n_z = 10
        self.z = core.Variable("z", np.random.normal(0.0, 2.0, (self.n_z, 1)))
        self.is_greater_than = core.Predicate.Lambda(lambda inputs: inputs[0] > inputs[1])
        self.add = core.Function.Lambda(lambda inputs: inputs[0] + inputs[1])
        self.mask1 = self.is_greater_than([self.x, self.y])
        self.mask2 = self.is_greater_than([self.add([self.x, self.y]), self.z])

        self.p1 = core.Predicate.MLP([1, 1])
        self.p2 = core.Predicate.MLP([1, 1, 1])

    def test_same_vars_in_wff(self):
        phi1 = self.p1([self.x, self.y])
        casted_phi1 = core.broadcast_wff_and_mask(phi1, self.mask1)
        self.assertEqual(casted_phi1.free_vars[: len(self.mask1.free_vars)], self.mask1.free_vars)
        take_x = np.random.randint(self.n_x)
        self.assertTrue(
            array_allclose(casted_phi1.take("x", take_x).tensor, phi1.take("x", take_x).tensor)
        )
        take_y = np.random.randint(self.n_y)
        self.assertTrue(
            array_allclose(casted_phi1.take("y", take_y).tensor, phi1.take("y", take_y).tensor)
        )

    def test_more_vars_in_wff(self):
        phi2 = self.p2([self.x, self.y, self.z])
        casted_phi2 = core.broadcast_wff_and_mask(phi2, self.mask1)
        self.assertEqual(casted_phi2.free_vars[: len(self.mask1.free_vars)], self.mask1.free_vars)
        take_x = np.random.randint(self.n_x)
        self.assertTrue(
            array_allclose(casted_phi2.take("x", take_x).tensor, phi2.take("x", take_x).tensor)
        )
        take_y = np.random.randint(self.n_y)
        self.assertTrue(
            array_allclose(casted_phi2.take("y", take_y).tensor, phi2.take("y", take_y).tensor)
        )

    def test_less_vars_in_wff(self):
        phi1 = self.p1([self.x, self.y])
        casted_phi1 = core.broadcast_wff_and_mask(phi1, self.mask2)
        self.assertEqual(casted_phi1.free_vars[: len(self.mask2.free_vars)], self.mask2.free_vars)
        take_x = np.random.randint(self.n_x)
        self.assertTrue(
            array_allclose(
                casted_phi1.take("x", take_x).take("z", 0).tensor, phi1.take("x", take_x).tensor
            )
        )
        take_y = np.random.randint(self.n_y)
        self.assertTrue(
            array_allclose(
                casted_phi1.take("y", take_y).take("z", 0).tensor, phi1.take("y", take_y).tensor
            )
        )


class TestGuardedQuantifier(unittest.TestCase):
    def setUp(self):
        self.n_x = 10
        self.x = core.Variable("x", np.random.normal(0.0, 1.0, (self.n_x, 1)))
        self.n_y = 10
        self.y = core.Variable("y", np.random.normal(0.0, 1.0, (self.n_y, 1)))
        self.is_greater_than = core.Predicate.Lambda(lambda inputs: inputs[0] > inputs[1])
        self.mask1 = self.is_greater_than([self.x, self.y])

        self.p1 = core.Predicate.MLP([1, 1])

        self.Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
        self.Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMean(p=5), semantics="exists")

    def test_values_correct(self):
        # pick an x that guards at least one y; the empty case is covered by test_empty_semantics_*
        non_empty_xs = np.flatnonzero(tf.reduce_any(self.mask1.tensor > 0, axis=1))
        take_x = np.random.choice(non_empty_xs)
        actual = self.Forall(self.y, self.p1([self.x, self.y]), mask=self.mask1).take("x", take_x)
        safe_y = core.Variable(
            "safe_y",
            tf.boolean_mask(self.y.tensor, self.x.take("x", take_x).tensor > self.y.tensor),
        )
        desired = self.Forall(safe_y, self.p1([self.x, safe_y])).take("x", take_x)
        self.assertTrue(array_allclose(actual.tensor, desired.tensor))

    def test_gradients(self):
        """Gradients reach the guarded truth values only, and stay finite when a guard is empty."""
        x = core.Variable("x", np.random.rand(3, 1))
        y = core.Variable("y", np.random.rand(4, 1))
        mask = core.Formula(
            tf.constant([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]]),
            free_vars=["x", "y"],
        )
        guarded = mask.tensor.numpy() > 0
        truth_values = tf.Variable(np.random.uniform(0.1, 0.9, (3, 4)), dtype=tf.float32)
        for quantifier in [self.Forall, self.Exists]:
            for var in [x, y]:  # aggregating y, the guard of the second x is empty
                with tf.GradientTape() as tape:
                    wff = core.Formula(tf.identity(truth_values), free_vars=["x", "y"])
                    res = quantifier(var, wff, mask=mask)
                grads = tape.gradient(res.tensor, truth_values).numpy()
                self.assertTrue(np.all(np.isfinite(grads)))
                self.assertTrue(np.all(grads[guarded] != 0.0))
                self.assertTrue(np.all(grads[~guarded] == 0.0))

    def test_empty_semantics_forall(self):
        x = core.Variable("x", np.random.rand(self.n_x, 1))
        y = core.Variable("y", np.random.rand(self.n_y, 1) + 1.0)  # all y are greater
        mask1 = self.is_greater_than([x, y])
        take_y = np.random.randint(self.n_y)
        actual = self.Forall(x, self.p1([x, y]), mask=mask1).take("y", take_y).tensor
        desired = 1.0
        self.assertEqual(actual, desired)

    def test_empty_semantics_exist(self):
        x = core.Variable("x", np.random.rand(self.n_x, 1))
        y = core.Variable("y", np.random.rand(self.n_y, 1) + 1.0)  # all y are greater
        mask1 = self.is_greater_than([x, y])
        take_y = np.random.randint(self.n_y)
        actual = self.Exists(x, self.p1([x, y]), mask=mask1).take("y", take_y).tensor
        desired = 0.0
        self.assertEqual(actual, desired)

    def test_aggreg_first_var_of_mask(self):
        x = core.Variable("x", np.random.rand(3, 1))
        mask = core.Formula(
            tf.constant([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
            free_vars=["x", "y"],
        )
        wff = core.Formula(
            tf.constant([[0.4, 0.2, 0.8, 0.0], [0.9, 0.1, 0.5, 0.2], [0.3, 0.0, 0.8, 0.9]]),
            free_vars=["x", "y"],
        )
        # Result after mask
        #       [[.4,.2,  ,  ],
        #        [  ,.1,.5,  ],
        #        [  ,  ,.8,  ]]),
        Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Mean(), semantics="exists")

        actual = Exists(x, wff, mask=mask)
        expected = np.array([0.4, 0.15, 0.65, 0.0])
        self.assertTrue(array_allclose(actual.tensor, expected))

    def test_wff_vars_not_in_mask(self):
        """The empty semantics broadcast over the wff's free variables that are not in the mask."""
        x = core.Variable("x", [[0.0], [1.0], [2.0]])
        y = core.Variable("y", [[0.0], [10.0], [20.0]])
        p = core.Predicate.Lambda(lambda inputs: tf.squeeze(inputs[2], axis=1))  # p(x,y,z) = z
        mask = core.Predicate.Lambda(lambda inputs: tf.squeeze(inputs[1] < 5.0, axis=1))(
            [x, y]
        )  # only y=0 guards x
        Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Min(), semantics="forall")
        Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Max(), semantics="exists")
        for z_values in [
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ]:  # as many z as y, and more z than y
            z = core.Variable("z", np.array(z_values)[:, np.newaxis])
            for quantifier, empty_value in [(Forall, 1.0), (Exists, 0.0)]:
                actual = quantifier(x, p([x, y, z]), mask=mask)
                expected = np.stack(
                    [z_values, [empty_value] * len(z_values), [empty_value] * len(z_values)]
                )
                self.assertEqual(actual.free_vars, ["y", "z"])
                self.assertTrue(array_allclose(actual.tensor, expected))

    def test_quantified_vars_not_in_mask(self):
        """A mask on other free variables leaves the whole domain to the quantified variables."""
        z = core.Variable("z", np.random.rand(2, 1))
        Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Min(), semantics="forall")
        Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Max(), semantics="exists")
        wffs_and_masks = [
            # the rows of the mask have different lengths; the first mask gave wrong values, the
            # second an error
            (
                np.random.rand(2, 3, 2),
                ["x", "y", "z"],
                [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                ["x", "y"],
            ),
            (
                np.random.rand(2, 3, 2),
                ["x", "y", "z"],
                [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                ["x", "y"],
            ),
            (np.random.rand(3, 2), ["x", "z"], [1.0, 0.0, 1.0], ["x"]),
        ]
        for wff_values, wff_vars, mask_values, mask_vars in wffs_and_masks:
            wff = core.Formula(tf.constant(wff_values, dtype=tf.float32), free_vars=wff_vars)
            mask = core.Formula(tf.constant(mask_values), free_vars=mask_vars)
            for quantifier, aggreg, empty_value in [(Forall, np.min, 1.0), (Exists, np.max, 0.0)]:
                with self.subTest(mask=mask_values, semantics=quantifier.semantics):
                    actual = quantifier(z, wff, mask=mask)
                    expected = np.where(
                        np.array(mask_values) > 0, aggreg(wff.tensor.numpy(), axis=-1), empty_value
                    )
                    self.assertEqual(actual.free_vars, mask_vars)
                    self.assertTrue(array_allclose(actual.tensor, expected))

    def test_mask_on_more_free_vars(self):
        """Quantifying one var of a mask on three vars keeps the other two.

        The aggregation result is ragged.
        """
        w = core.Variable("w", np.random.rand(4, 1))
        wff_values = np.random.rand(2, 3, 4)
        mask_values = np.random.rand(2, 3, 4) > 0.5
        mask_values[0, 0] = False  # an empty guard
        wff = core.Formula(tf.constant(wff_values, dtype=tf.float32), free_vars=["x", "y", "w"])
        mask = core.Formula(tf.constant(mask_values, dtype=tf.float32), free_vars=["x", "y", "w"])
        Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Min(), semantics="forall")
        actual = Forall(w, wff, mask=mask)
        expected = np.where(mask_values, wff_values, np.inf).min(axis=2)
        expected[~mask_values.any(axis=2)] = 1.0
        self.assertEqual(actual.free_vars, ["x", "y"])
        self.assertTrue(array_allclose(actual.tensor, expected))


class TestDiag(unittest.TestCase):
    def setUp(self):
        self.x = core.Variable("x", np.random.rand(3, 1))
        self.y = core.Variable("y", np.random.rand(3, 1))
        self.p = core.Predicate.Lambda(lambda inputs: tf.squeeze(inputs[0] * inputs[1], axis=1))
        self.Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Mean(), semantics="forall")

    def test_diag_pairs_individuals(self):
        x, y = core.diag(self.x, self.y)
        phi = self.p([x, y])
        self.assertEqual(phi.tensor.shape, [3])
        self.assertTrue(array_allclose(phi.tensor, self.x.tensor[:, 0] * self.y.tensor[:, 0]))

    def test_quantifier_undiags(self):
        core.diag(self.x, self.y)
        self.Forall([self.x, self.y], self.p([self.x, self.y]))
        self.assertEqual(self.x.free_vars, ["x"])
        self.assertEqual(self.y.free_vars, ["y"])

    def test_diag_lock_survives_quantifier(self):
        core.diag_lock(self.x, self.y)
        self.Forall([self.x, self.y], self.p([self.x, self.y]))
        self.assertEqual(self.x.free_vars, ["diaglock_x_y"])
        self.assertEqual(self.y.free_vars, ["diaglock_x_y"])
        self.assertEqual(self.p([self.x, self.y]).tensor.shape, [3])

    def test_error_messages_name_the_variable(self):
        core.diag(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "temporarily diagged: x"):
            core.diag(self.x, self.y)
        with self.assertRaisesRegex(ValueError, "temporarily diagged: x"):
            core.diag_lock(self.x, self.y)


class TestInTfFunction(unittest.TestCase):
    def setUp(self):
        self.x = core.Variable("x", np.random.rand(3, 2))
        self.y = core.Variable("y", np.random.rand(4, 2))
        self.p = core.Predicate.MLP([2, 2])
        self.Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")

    def test_take_with_tensor_indices(self):
        @tf.function
        def take(indices):
            return self.p([self.x, self.y]).take("x", indices).tensor

        phi = self.p([self.x, self.y])
        self.assertTrue(array_allclose(take(tf.constant(1)), phi.take("x", 1).tensor))
        self.assertTrue(array_allclose(take(tf.constant([0, 2])), phi.take("x", [0, 2]).tensor))

    def test_quantifier(self):
        @tf.function
        def sat():
            return self.Forall([self.x, self.y], self.p([self.x, self.y])).tensor

        self.assertTrue(
            array_allclose(sat(), self.Forall([self.x, self.y], self.p([self.x, self.y])).tensor)
        )


class TestFormulaAggregator(unittest.TestCase):
    def test_passes_kwargs(self):
        aggregator = core.Wrapper_Formula_Aggregator(fuzzy_ops.Aggreg_pMean(p=2))
        wffs = [core.Proposition(0.0, trainable=False), core.Proposition(1.0, trainable=False)]
        self.assertTrue(array_allclose(aggregator(wffs, p=1, stable=False).tensor, np.array(0.5)))

    def test_rejects_free_vars(self):
        aggregator = core.Wrapper_Formula_Aggregator(fuzzy_ops.Aggreg_Mean())
        phi = core.Predicate.MLP([2])(core.Variable("x", np.random.rand(3, 2)))
        with self.assertRaises(ValueError):
            aggregator([phi])


class TestOptimizers(unittest.TestCase):
    """Trainable constants and propositions are `tf.Variable`s, not Keras variables:
    Keras optimizers must update them too (see the smokes-friends-cancer example)."""

    def test_constants(self):
        for optimizer in [
            tf.keras.optimizers.SGD(learning_rate=0.1),
            tf.keras.optimizers.Adam(learning_rate=0.1),
        ]:
            with self.subTest(optimizer=optimizer.__class__.__name__):
                c1 = core.Constant([0.0, 0.0], trainable=True)
                c2 = core.Constant([0.0, 0.0], trainable=True)
                c3 = core.Constant(0.0, trainable=True)
                P = core.Predicate.Lambda(lambda x: tf.sigmoid(tf.reduce_sum(x, axis=1)))
                Forall = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Mean(), semantics="forall")
                trainable_variables = [c1.tensor, c2.tensor, c3.tensor]
                with tf.GradientTape() as tape:
                    x = core.Variable.from_constants("x", [c1, c2], tape=tape)
                    loss = 2.0 - Forall(x, P(x)).tensor - P(c3).tensor
                optimizer.apply_gradients(
                    zip(tape.gradient(loss, trainable_variables), trainable_variables, strict=True)
                )
                # P increases with the features of its argument
                for c in [c1, c2, c3]:
                    self.assertTrue(np.all(c.tensor.numpy() > 0.0))

    def test_propositions_stay_in_truth_range(self):
        And = core.Wrapper_Connective(fuzzy_ops.And_Prod())
        Not = core.Wrapper_Connective(fuzzy_ops.Not_Std())
        for optimizer in [
            tf.keras.optimizers.SGD(learning_rate=5.0),
            tf.keras.optimizers.Adam(learning_rate=5.0),
        ]:
            with self.subTest(optimizer=optimizer.__class__.__name__):
                a = core.Proposition(0.5, trainable=True)
                b = core.Proposition(0.5, trainable=True)
                trainable_variables = [a.tensor, b.tensor]
                with tf.GradientTape() as tape:
                    loss = 1.0 - And(a, Not(b)).tensor
                optimizer.apply_gradients(
                    zip(tape.gradient(loss, trainable_variables), trainable_variables, strict=True)
                )
                # the large step pushes the truth values out of [0,1], and the constraint clips them
                # back
                self.assertEqual(float(a.tensor), 1.0)
                self.assertEqual(float(b.tensor), 0.0)


class TestDigitAdditionAxiom(unittest.TestCase):
    """The axiom of the MNIST addition example combines diag, a mask built with a function,
    and a predicate from logits. Compare it with a direct computation."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.n_digits = 3
        self.images_x = rng.normal(size=(5, 4)).astype(np.float32)
        self.images_y = rng.normal(size=(5, 4)).astype(np.float32)
        self.labels = np.array(
            [0.0, 1.0, 3.0, 4.0, 5.0]
        )  # no two digits sum to 5: the guard of the last pair is empty
        self.logits_model = ListInputLogits(self.n_digits)
        self.logits_model([self.images_x])  # builds the weights
        self.Digit = core.Predicate.FromLogits(self.logits_model, activation_function="softmax")
        self.d1 = core.Variable("d1", range(self.n_digits))
        self.d2 = core.Variable("d2", range(self.n_digits))
        self.add = core.Function.Lambda(lambda inputs: inputs[0] + inputs[1])
        self.equals = core.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])
        self.And = core.Wrapper_Connective(fuzzy_ops.And_Prod(stable=False))
        self.Forall = core.Wrapper_Quantifier(
            fuzzy_ops.Aggreg_pMeanError(p=2, stable=False), semantics="forall"
        )
        self.Exists = core.Wrapper_Quantifier(
            fuzzy_ops.Aggreg_pMean(p=2, stable=False), semantics="exists"
        )

    def axiom(self):
        images_x = core.Variable("x", self.images_x)
        images_y = core.Variable("y", self.images_y)
        labels_z = core.Variable("z", self.labels)
        return self.Forall(
            core.diag(images_x, images_y, labels_z),
            self.Exists(
                (self.d1, self.d2),
                self.And(self.Digit([images_x, self.d1]), self.Digit([images_y, self.d2])),
                mask=self.equals([self.add([self.d1, self.d2]), labels_z]),
            ),
        ).tensor

    def expected(self):
        probs_x = tf.nn.softmax(self.logits_model([self.images_x])).numpy()
        probs_y = tf.nn.softmax(self.logits_model([self.images_y])).numpy()
        sat_per_pair = []
        for p_x, p_y, label in zip(probs_x, probs_y, self.labels, strict=True):
            products = [
                p_x[d1] * p_y[d2]
                for d1 in range(self.n_digits)
                for d2 in range(self.n_digits)
                if d1 + d2 == label
            ]
            sat_per_pair.append(np.sqrt(np.mean(np.square(products))) if products else 0.0)
        return 1.0 - np.sqrt(np.mean(np.square(1.0 - np.array(sat_per_pair))))

    def test_values_correct(self):
        self.assertTrue(np.isclose(float(self.axiom()), self.expected(), atol=1e-5))

    def test_in_tf_function(self):
        self.assertTrue(np.isclose(float(tf.function(self.axiom)()), self.expected(), atol=1e-5))


class TestUtils(unittest.TestCase):
    def test_smooth_equality_gradients_at_equal_inputs(self):
        for pred in [ltn.utils.Pred_equal_smooth_exp(), ltn.utils.Pred_equal_smooth_inv()]:
            u = tf.Variable([[1.0, 2.0], [0.0, 0.0]])
            v = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            with tf.GradientTape() as tape:
                out = pred.model([u, v])
            self.assertTrue(np.all(np.isfinite(tape.gradient(out, u).numpy())))
            self.assertTrue(array_allclose(out[0], np.float32(1.0)))

    def test_smooth_equality_values(self):
        u, v = tf.constant([[0.0, 0.0]]), tf.constant([[3.0, 4.0]])  # distance 5
        self.assertTrue(
            array_allclose(ltn.utils.Pred_equal_smooth_exp(alpha=0.1).model([u, v]), np.exp([-0.5]))
        )
        self.assertTrue(
            array_allclose(
                ltn.utils.Pred_equal_smooth_inv(alpha=0.1).model([u, v]), np.array([1 / 1.5])
            )
        )


class TestTypeCheck(unittest.TestCase):
    def setUp(self):
        self.x = core.Variable("x", np.random.rand(3, 1))
        self.y = core.Variable("y", np.random.rand(4, 1))
        self.c = core.Constant([3.0], trainable=False)
        self.f1 = core.Function.MLP(input_shapes=[1], output_shape=[1])
        self.f2 = core.Function.MLP(input_shapes=[1, 1], output_shape=[1])
        self.p1 = core.Predicate.MLP(input_shapes=[1])
        self.p2 = core.Predicate.MLP(input_shapes=[1, 1])
        self.q = core.Proposition(0.0, trainable=False)
        self.And = core.Wrapper_Connective(fuzzy_ops.And_Prod())
        self.Not = core.Wrapper_Connective(fuzzy_ops.Not_Std())
        self.Exists = core.Wrapper_Quantifier(fuzzy_ops.Aggreg_Mean(), semantics="exists")
        self.mask = core.Formula(
            tf.constant([[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
            free_vars=["x", "y"],
        )

    def test_predicate(self):
        try:
            self.p1(self.x)
            self.p2([self.x, self.c])
            self.p1(self.f1(self.x))
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.p1(self.x.tensor)
        with self.assertRaises(TypeError):
            self.p2([self.x, self.p1(self.x)])

    def test_function(self):
        try:
            self.f1(self.x)
            self.f2([self.x, self.c])
            self.f1(self.f1(self.x))
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.f1(self.x.tensor)
        with self.assertRaises(TypeError):
            self.f2([self.x, self.p1(self.x)])

    def test_connective(self):
        try:
            self.And(self.p1(self.x), self.q)
            self.Not(self.p2([self.x, self.c]))
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.And(self.q, self.f1(self.x))
        with self.assertRaises(TypeError):
            self.Not(self.p1(self.x).tensor)

    def test_quantifier(self):
        try:
            self.Exists(self.x, self.p1(self.x))
            self.Exists((self.x, self.y), self.p2([self.x, self.y]))
            self.Exists((self.x, self.y), self.p2([self.x, self.y]), mask=self.mask)
        except TypeError:
            self.fail("TypeError raised unexpectedly.")
        with self.assertRaises(TypeError):
            self.Exists(self.q, self.p1(self.x))
        with self.assertRaisesRegex(TypeError, "instance of <class 'ltn.core.Term'>"):
            self.Exists(self.x, self.f1(self.x))
        with self.assertRaises(TypeError):
            self.Exists((self.x, self.y), self.p2([self.x, self.y]), mask=self.mask.tensor)
