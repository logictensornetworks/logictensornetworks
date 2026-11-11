import unittest
import warnings

import numpy as np
import tensorflow as tf

from ltn import fuzzy_ops

# grid of truth values, including the boundaries 0 and 1
VALUES = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], dtype=np.float32)
X, Y = [tf.constant(t.flatten()) for t in np.meshgrid(VALUES, VALUES)]
ZEROS, ONES = tf.zeros_like(X), tf.ones_like(X)

# the stable variants deviate from the exact operators by at most ~eps
ATOL = 10 * fuzzy_ops.eps


def allclose(actual, desired, atol=ATOL):
    return np.allclose(np.asarray(actual), np.asarray(desired), atol=atol)


NOTS = [fuzzy_ops.Not_Std(), fuzzy_ops.Not_Godel()]
ANDS = [
    fuzzy_ops.And_Min(),
    fuzzy_ops.And_Prod(),
    fuzzy_ops.And_Prod(stable=False),
    fuzzy_ops.And_Luk(),
]
ORS = [
    fuzzy_ops.Or_Max(),
    fuzzy_ops.Or_ProbSum(),
    fuzzy_ops.Or_ProbSum(stable=False),
    fuzzy_ops.Or_Luk(),
]
IMPLIES = [
    fuzzy_ops.Implies_KleeneDienes(),
    fuzzy_ops.Implies_Godel(),
    fuzzy_ops.Implies_Reichenbach(),
    fuzzy_ops.Implies_Reichenbach(stable=False),
    fuzzy_ops.Implies_Goguen(),
    fuzzy_ops.Implies_Goguen(stable=False),
    fuzzy_ops.Implies_Luk(),
]
# implications that are the residuum of a t-norm: I(x,y) = 1 iff x <= y
RESIDUAL_IMPLIES = [
    fuzzy_ops.Implies_Godel(),
    fuzzy_ops.Implies_Goguen(stable=False),
    fuzzy_ops.Implies_Luk(),
]


def name(op):
    stable = getattr(op, "stable", None)
    return op.__class__.__name__ + ("" if stable is None else f"(stable={stable})")


class TestNegation(unittest.TestCase):
    def test_range(self):
        for op in NOTS:
            with self.subTest(op=name(op)):
                out = op(X).numpy()
                self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_boundaries(self):
        for op in NOTS:
            with self.subTest(op=name(op)):
                self.assertTrue(allclose(op(tf.constant([0.0, 1.0])), [1.0, 0.0]))

    def test_std_is_involutive(self):
        self.assertTrue(allclose(fuzzy_ops.Not_Std()(fuzzy_ops.Not_Std()(X)), X, atol=1e-6))

    def test_godel_is_crisp(self):
        expected = (VALUES == 0.0).astype(np.float32)
        self.assertTrue(allclose(fuzzy_ops.Not_Godel()(tf.constant(VALUES)), expected, atol=0.0))

    def test_monotonically_decreasing(self):
        for op in NOTS:
            with self.subTest(op=name(op)):
                self.assertTrue(np.all(np.diff(op(tf.constant(VALUES)).numpy()) <= 0.0))


class TestConjunctionDisjunction(unittest.TestCase):
    def test_range(self):
        for op in ANDS + ORS:
            with self.subTest(op=name(op)):
                out = op(X, Y).numpy()
                self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_commutative(self):
        for op in ANDS + ORS:
            with self.subTest(op=name(op)):
                self.assertTrue(allclose(op(X, Y), op(Y, X), atol=1e-6))

    def test_and_neutral_and_absorbing_elements(self):
        for op in ANDS:
            with self.subTest(op=name(op)):
                self.assertTrue(allclose(op(X, ONES), X))
                self.assertTrue(allclose(op(X, ZEROS), ZEROS))

    def test_or_neutral_and_absorbing_elements(self):
        for op in ORS:
            with self.subTest(op=name(op)):
                self.assertTrue(allclose(op(X, ZEROS), X))
                self.assertTrue(allclose(op(X, ONES), ONES))

    def test_and_below_min_and_or_above_max(self):
        for op in ANDS:
            with self.subTest(op=name(op)):
                self.assertTrue(np.all(op(X, Y).numpy() <= np.minimum(X, Y) + ATOL))
        for op in ORS:
            with self.subTest(op=name(op)):
                self.assertTrue(np.all(op(X, Y).numpy() >= np.maximum(X, Y) - ATOL))

    def test_monotone(self):
        # increasing the first argument never decreases the result
        x_low, x_high = tf.constant(VALUES[:-1]), tf.constant(VALUES[1:])
        for op in ANDS + ORS:
            for y in VALUES:
                y_ = tf.fill(x_low.shape, y)
                with self.subTest(op=name(op), y=y):
                    self.assertTrue(np.all(op(x_low, y_).numpy() <= op(x_high, y_).numpy() + 1e-6))

    def test_de_morgan(self):
        Not = fuzzy_ops.Not_Std()
        pairs = [
            (fuzzy_ops.And_Min(), fuzzy_ops.Or_Max()),
            (fuzzy_ops.And_Prod(stable=False), fuzzy_ops.Or_ProbSum(stable=False)),
            (fuzzy_ops.And_Luk(), fuzzy_ops.Or_Luk()),
        ]
        for And, Or in pairs:
            with self.subTest(And=name(And), Or=name(Or)):
                self.assertTrue(allclose(Or(X, Y), Not(And(Not(X), Not(Y))), atol=1e-6))

    def test_associative(self):
        Z = tf.reverse(X, axis=[0])
        for op in [
            fuzzy_ops.And_Min(),
            fuzzy_ops.And_Prod(stable=False),
            fuzzy_ops.And_Luk(),
            fuzzy_ops.Or_Max(),
            fuzzy_ops.Or_ProbSum(stable=False),
            fuzzy_ops.Or_Luk(),
        ]:
            with self.subTest(op=name(op)):
                self.assertTrue(allclose(op(op(X, Y), Z), op(X, op(Y, Z)), atol=1e-6))

    def test_broadcasting(self):
        x = tf.constant([[0.2], [0.8]])
        y = tf.constant([0.1, 0.5, 0.9])
        for op in ANDS + ORS:
            with self.subTest(op=name(op)):
                self.assertEqual(op(x, y).shape, [2, 3])


class TestImplication(unittest.TestCase):
    def test_range(self):
        for op in IMPLIES:
            with self.subTest(op=name(op)):
                out = op(X, Y).numpy()
                self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_boundaries(self):
        for op in IMPLIES:
            with self.subTest(op=name(op)):
                # ex falso quodlibet. The stable Goguen implication shifts x=0 to eps,
                # so it only holds for y >= eps (it returns y/eps below that).
                y = Y[fuzzy_ops.eps <= Y]
                self.assertTrue(allclose(op(tf.zeros_like(y), y), tf.ones_like(y)))
                self.assertTrue(allclose(op(X, ONES), ONES))
                self.assertTrue(allclose(op(ONES, Y), Y))  # left neutrality

    def test_monotonicity(self):
        # decreasing in the antecedent, increasing in the consequent
        low, high = tf.constant(VALUES[:-1]), tf.constant(VALUES[1:])
        for op in IMPLIES:
            for v in VALUES:
                v_ = tf.fill(low.shape, v)
                with self.subTest(op=name(op), v=v):
                    self.assertTrue(np.all(op(low, v_).numpy() >= op(high, v_).numpy() - 1e-6))
                    self.assertTrue(np.all(op(v_, low).numpy() <= op(v_, high).numpy() + 1e-6))

    def test_residual_implications_order_property(self):
        for op in RESIDUAL_IMPLIES:
            with self.subTest(op=name(op)):
                out = op(X, Y).numpy()
                self.assertTrue(np.all(np.isclose(out, 1.0) == (X.numpy() <= Y.numpy())))

    def test_s_implications(self):
        # Kleene-Dienes and Reichenbach are I(x,y) = Or(Not(x),y)
        Not = fuzzy_ops.Not_Std()
        self.assertTrue(
            allclose(
                fuzzy_ops.Implies_KleeneDienes()(X, Y), fuzzy_ops.Or_Max()(Not(X), Y), atol=1e-6
            )
        )
        self.assertTrue(
            allclose(
                fuzzy_ops.Implies_Reichenbach(stable=False)(X, Y),
                fuzzy_ops.Or_ProbSum(stable=False)(Not(X), Y),
                atol=1e-6,
            )
        )

    def test_residuum_of_t_norm(self):
        # And(x,I(x,y)) <= y (modus ponens) for residual implications and their t-norm
        pairs = [
            (fuzzy_ops.And_Min(), fuzzy_ops.Implies_Godel()),
            (fuzzy_ops.And_Prod(stable=False), fuzzy_ops.Implies_Goguen(stable=False)),
            (fuzzy_ops.And_Luk(), fuzzy_ops.Implies_Luk()),
        ]
        for And, Implies in pairs:
            with self.subTest(And=name(And), Implies=name(Implies)):
                self.assertTrue(np.all(And(X, Implies(X, Y)).numpy() <= Y.numpy() + 1e-6))


class TestEquiv(unittest.TestCase):
    def setUp(self):
        self.equivs = [
            fuzzy_ops.Equiv(fuzzy_ops.And_Min(), fuzzy_ops.Implies_Godel()),
            fuzzy_ops.Equiv(fuzzy_ops.And_Prod(), fuzzy_ops.Implies_Reichenbach()),
            fuzzy_ops.Equiv(fuzzy_ops.And_Luk(), fuzzy_ops.Implies_Luk()),
        ]

    def test_symmetric(self):
        for op in self.equivs:
            with self.subTest(op=name(op.implies_op)):
                self.assertTrue(allclose(op(X, Y), op(Y, X), atol=1e-6))

    def test_crisp_values(self):
        x, y = tf.constant([0.0, 0.0, 1.0, 1.0]), tf.constant([0.0, 1.0, 0.0, 1.0])
        for op in self.equivs:
            with self.subTest(op=name(op.implies_op)):
                self.assertTrue(allclose(op(x, y), [1.0, 0.0, 0.0, 1.0]))

    def test_definition(self):
        And, Implies = fuzzy_ops.And_Luk(), fuzzy_ops.Implies_Luk()
        self.assertTrue(
            allclose(
                fuzzy_ops.Equiv(And, Implies)(X, Y), And(Implies(X, Y), Implies(Y, X)), atol=0.0
            )
        )

    def test_reflexive_for_residual_implications(self):
        for op in [self.equivs[0], self.equivs[2]]:
            with self.subTest(op=name(op.implies_op)):
                self.assertTrue(allclose(op(X, X), ONES, atol=0.0))


class TestAggregators(unittest.TestCase):
    def setUp(self):
        self.xs = tf.constant(np.random.default_rng(0).uniform(size=(4, 5)), dtype=tf.float32)
        self.aggregators = [
            fuzzy_ops.Aggreg_Min(),
            fuzzy_ops.Aggreg_Max(),
            fuzzy_ops.Aggreg_Mean(),
            fuzzy_ops.Aggreg_pMean(),
            fuzzy_ops.Aggreg_pMeanError(),
            fuzzy_ops.Aggreg_Prod(),
        ]

    def test_range(self):
        for op in self.aggregators:
            with self.subTest(op=name(op)):
                out = op(self.xs, axis=1).numpy()
                self.assertTrue(np.all((out >= 0.0) & (out <= 1.0)))

    def test_idempotent(self):
        for c in [0.0, 0.3, 1.0]:
            xs = tf.fill([6], c)
            for op in self.aggregators[:-1]:  # the product is not idempotent
                with self.subTest(op=name(op), c=c):
                    self.assertTrue(allclose(op(xs), c))

    def test_ordering(self):
        # min <= pMeanError <= mean <= pMean <= max
        ops = [
            fuzzy_ops.Aggreg_Min(),
            fuzzy_ops.Aggreg_pMeanError(p=2, stable=False),
            fuzzy_ops.Aggreg_Mean(),
            fuzzy_ops.Aggreg_pMean(p=2, stable=False),
            fuzzy_ops.Aggreg_Max(),
        ]
        results = np.stack([op(self.xs, axis=1).numpy() for op in ops])
        self.assertTrue(np.all(np.diff(results, axis=0) >= -1e-6))

    def test_p_controls_strictness(self):
        # pMean tends to max and pMeanError tends to min as p grows
        means = [
            fuzzy_ops.Aggreg_pMean(stable=False)(self.xs, axis=1, p=p).numpy()
            for p in [1, 2, 5, 20]
        ]
        errors = [
            fuzzy_ops.Aggreg_pMeanError(stable=False)(self.xs, axis=1, p=p).numpy()
            for p in [1, 2, 5, 20]
        ]
        self.assertTrue(np.all(np.diff(means, axis=0) >= -1e-6))
        self.assertTrue(np.all(np.diff(errors, axis=0) <= 1e-6))
        mean = tf.reduce_mean(self.xs, axis=1)
        self.assertTrue(allclose(means[0], mean, atol=1e-6))
        self.assertTrue(allclose(errors[0], mean, atol=1e-6))

    def test_values(self):
        xs = tf.constant([0.2, 0.4, 0.8])
        self.assertTrue(
            allclose(
                fuzzy_ops.Aggreg_pMean(p=2, stable=False)(xs),
                np.sqrt((0.04 + 0.16 + 0.64) / 3),
                atol=1e-6,
            )
        )
        self.assertTrue(
            allclose(
                fuzzy_ops.Aggreg_pMeanError(p=2, stable=False)(xs),
                1 - np.sqrt((0.64 + 0.36 + 0.04) / 3),
                atol=1e-6,
            )
        )
        self.assertTrue(allclose(fuzzy_ops.Aggreg_Prod()(xs), 0.2 * 0.4 * 0.8, atol=1e-6))

    def test_axis_and_keepdims(self):
        xs = tf.ones([2, 3, 4])
        for op in self.aggregators:
            with self.subTest(op=name(op)):
                self.assertEqual(op(xs).shape, [])
                self.assertEqual(op(xs, axis=1).shape, [2, 4])
                self.assertEqual(op(xs, axis=[0, 2]).shape, [3])
                self.assertEqual(op(xs, axis=1, keepdims=True).shape, [2, 1, 4])

    def test_ragged_tensors(self):
        # guarded quantifiers aggregate ragged tensors
        xs = tf.ragged.constant([[0.2, 0.4], [0.6], [0.1, 0.5, 0.9]])
        for op in self.aggregators:
            with self.subTest(op=name(op)):
                out = op(xs, axis=1)
                self.assertEqual(out.shape, [3])


class TestLogProd(unittest.TestCase):
    def test_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fuzzy_ops.Aggreg_LogProd()
        self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))

    def test_values(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op = fuzzy_ops.Aggreg_LogProd(stable=False)
        xs = tf.constant([[0.2, 0.4], [0.5, 1.0]])
        self.assertTrue(allclose(op(xs, axis=1), np.log([0.08, 0.5]), atol=1e-6))
        self.assertTrue(allclose(op(xs), np.log(0.04), atol=1e-6))

    def test_stable_is_finite_at_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op = fuzzy_ops.Aggreg_LogProd()
        self.assertTrue(np.isfinite(op(tf.constant([0.0, 0.5])).numpy()))
        self.assertFalse(np.isfinite(op(tf.constant([0.0, 0.5]), stable=False).numpy()))


class TestStableGradients(unittest.TestCase):
    """The stable variants exist to avoid NaN/inf gradients at the boundaries 0 and 1."""

    def assert_finite_gradients(self, fn, n_args):
        args = [tf.Variable(X), tf.Variable(Y)][:n_args]
        with tf.GradientTape() as tape:
            out = tf.reduce_sum(fn(*args))
        for grad in tape.gradient(out, args):
            self.assertTrue(np.all(np.isfinite(grad.numpy())))

    def test_binary_operators(self):
        for op in [
            fuzzy_ops.And_Prod(),
            fuzzy_ops.Or_ProbSum(),
            fuzzy_ops.Implies_Reichenbach(),
            fuzzy_ops.Implies_Goguen(),
        ]:
            with self.subTest(op=name(op)):
                self.assert_finite_gradients(op, 2)

    def test_aggregators(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_prod = fuzzy_ops.Aggreg_LogProd()
        for op in [fuzzy_ops.Aggreg_pMean(p=2), fuzzy_ops.Aggreg_pMeanError(p=2), log_prod]:
            for xs in [ZEROS, ONES]:
                with self.subTest(op=name(op), xs=float(xs[0])):
                    self.assert_finite_gradients(lambda x: op(x * xs), 1)  # noqa: B023 (called right away)

    def test_unstable_aggregators_are_not_finite(self):
        """Sanity check that the boundaries above do exercise the instability."""
        cases = [
            (fuzzy_ops.Aggreg_pMean(p=2, stable=False), ZEROS),
            (fuzzy_ops.Aggreg_pMeanError(p=2, stable=False), ONES),
        ]
        for op, xs in cases:
            with self.subTest(op=name(op)):
                x = tf.Variable(xs)
                with tf.GradientTape() as tape:
                    out = op(x)
                self.assertFalse(np.all(np.isfinite(tape.gradient(out, x).numpy())))


if __name__ == "__main__":
    unittest.main()
