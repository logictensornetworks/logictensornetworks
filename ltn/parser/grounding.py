"""Turns a syntax tree into LTN objects. It doesn't depend on the syntax the tree was read from.

Formulas are parsed once into a syntax tree. Every call to `Grounding.sat()` evaluates the tree
again, so the LTN variables built from trainable constants are created while the caller's
GradientTape is recording.
"""
from __future__ import annotations

import functools
import itertools
import sys
import warnings

import tensorflow as tf

import ltn
from ltn.parser.tree import (
    AXIOM_ROLES,
    Atom,
    BinOp,
    Chain,
    Fn,
    Not,
    Problem,
    Quant,
    Statement,
    Truth,
    Var,
)


def smooth_equality(alpha: float = 1.0) -> ltn.Predicate:
    """exp(-alpha * squared distance). Unlike exp(-alpha * distance), its gradient is
    defined when both arguments are equal."""
    return ltn.Predicate.Lambda(
        lambda args: tf.exp(-alpha * tf.reduce_sum(tf.square(args[0] - args[1]), axis=1)))


class Grounding:
    """The default grounding of a problem: constants are trainable embeddings, functions and
    predicates are MLPs, propositions are trainable truth values. Variables range over the
    individuals in `domain` and, with `herbrand_depth` > 0, over the functions applied to them up
    to that depth. Every part of it can be replaced before training.

    Constants that the problem interprets as themselves, such as TPTP's numbers and distinct
    objects, are one individual if their values are equal, so they share a grounding. An equation
    between two of them is true or false according to their values, and `sat` includes the
    implicit axiom that different values are unequal."""

    def __init__(self, problem: Problem, embedding_size: int = 8, extra_individuals: int = 0,
                 herbrand_depth: int = 0) -> None:
        self.problem = problem
        self.herbrand_depth = herbrand_depth
        d = embedding_size
        names = [f for f, arity in problem.functions.items() if arity == 0]
        names += [f"_individual{i}" for i in range(extra_individuals)]
        self.constants, self.domain, by_value = {}, [], {}
        for name in names:
            value = problem.values.get(name)
            if value in by_value:  # an equal value is the same individual
                self.constants[name] = by_value[value]
                continue
            self.constants[name] = ltn.Constant(tf.random.uniform([d]), trainable=True)
            self.domain.append(name)
            if value is not None:
                by_value[value] = self.constants[name]
        self.functions = {
            f: ltn.Function.MLP([d] * n, d) for f, n in problem.functions.items() if n > 0
        }
        self.predicates = {
            p: ltn.Predicate.MLP([d] * n) for p, n in problem.predicates.items() if n > 0
        }
        self.predicates["="] = smooth_equality()
        self.propositions = {p: ltn.Proposition(0.5, trainable=True)
                             for p, n in problem.predicates.items() if n == 0}
        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.connectives = {
            "&": ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod()),
            "|": ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum()),
            "=>": ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach()),
            "<=>": ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(
                ltn.fuzzy_ops.And_Prod(), ltn.fuzzy_ops.Implies_Reichenbach())),
        }
        self.quantifiers = {
            "!": ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall"),
            "?": ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6), semantics="exists"),
        }
        self.aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        params = [c.tensor for c in self.constants.values()]
        params += [p.tensor for p in self.propositions.values()]
        for symbol in itertools.chain(self.functions.values(), self.predicates.values()):
            params += symbol.model.trainable_variables
        # constants with equal values share one grounding, so a parameter can come twice
        return list({id(param): param for param in params}.values())

    def individuals(self) -> list[ltn.core.Term]:
        """The individuals that variables range over. Call it inside the tape when training:
        they depend on the trainable constants and functions."""
        terms = {Fn(c): self.constants[c] for c in self.domain}
        for _ in range(self.herbrand_depth):
            new = {}
            for f, function in self.functions.items():
                for args in itertools.product(list(terms), repeat=self.problem.functions[f]):
                    if Fn(f, args) not in terms:
                        new[Fn(f, args)] = self._apply(function, [terms[a] for a in args])
            terms.update(new)
        return list(terms.values())

    def evaluate(self, formula, tape: tf.GradientTape | None = None,
                 individuals: list[ltn.core.Term] | None = None) -> ltn.core.Formula:
        """Builds the LTN formula. Call it inside the tape when training."""
        if isinstance(formula, str):
            formula = self.problem.parse_formula(formula)
        if individuals is None:
            individuals = self.individuals()
        try:
            return self._eval(formula, {}, lambda name: self._variable(name, individuals, tape))
        except RecursionError:
            raise NotImplementedError(
                f"The formula is nested too deeply for Python's recursion limit "
                f"({sys.getrecursionlimit()})."
            ) from None

    def sat(
        self, statements: list[Statement] | None = None, tape: tf.GradientTape | None = None
    ) -> tf.Tensor:
        """The satisfaction of `statements`, by default the axioms, as one truth value. It includes
        the implicit axiom that different values are unequal, and it is 1 if there is nothing to
        satisfy."""
        if statements is None:
            statements = self.problem.with_role(*AXIOM_ROLES)
        individuals = self.individuals()
        formulas = [self.evaluate(s.formula, tape, individuals) for s in statements]
        distinct = self._distinct_values()
        if distinct is not None:
            formulas.append(distinct)
        if not formulas:
            return tf.constant(1.)
        return self.aggregator(formulas).tensor

    def train(self, steps: int, optimizer: tf.keras.optimizers.Optimizer | None = None,
              reuse_graph: bool = False, statements: list[Statement] | None = None) -> float:
        """Maximizes the satisfaction of `statements`, by default the axioms, as `sat` computes it.
        Returns the last satisfaction. A CNF problem usually puts its conjecture in
        negated_conjecture clauses, which are not axioms: to train on them too, pass
        `statements=problem.statements`.

        By default each call builds a new graph, and a new Adam(0.01) if no optimizer is given. If
        you call this repeatedly, for example to log between calls, `reuse_graph=True` saves the
        time it takes to build the graph: the first call stores its graph and its optimizer on this
        object, and later calls with the flag reuse both. If a later call with the flag passes a
        different optimizer object, it builds a new graph with that one, because a graph is tied to
        its optimizer. `optimizer=None` keeps the stored one.

        The flag changes the results, not only the speed. Each call without it starts a new Adam,
        so `train(5)` twice doesn't match `train(10)`. With the flag the optimizer's state carries
        on and the two match exactly. Passing the same optimizer to two calls without the flag also
        matches `train(10)`.

        Only use the flag if you are sure that the graph hasn't changed. A kept graph picks up new
        values of constants, propositions and network weights, through `assign` or training, and
        the optimizer's learning rate, for example `optimizer.learning_rate = 0.5`. It silently
        misses anything else and keeps using the old objects, with no error: the formulas, a new
        network or constant object for a symbol, the domain, a new connective or quantifier, and
        `p` inside a quantifier (about 0.8 in the kept graph and 0.5 in a new one, in one test).
        To replace a kept graph after such a change, pass a different optimizer. A kept graph is
        also tied to its statements: a call with different statements builds a new graph."""
        kept_step, kept_optimizer, kept_statements = getattr(self, "_kept_step", (None, None, None))
        if reuse_graph:
            optimizer = optimizer or kept_optimizer
        optimizer = optimizer or tf.keras.optimizers.Adam(0.01)
        params = self.trainable_variables
        key = None if statements is None else tuple(statements)

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                sat = self.sat(statements, tape=tape)
                loss = 1. - sat
            grads = tape.gradient(loss, params)
            pairs = [(g, p) for g, p in zip(grads, params, strict=True) if g is not None]
            if pairs:  # without axioms, for example, there is nothing to train
                optimizer.apply_gradients(pairs)
            return sat

        if reuse_graph and optimizer is kept_optimizer and key == kept_statements:
            # the new tf.function costs nothing: it has no graph until its first call
            step = kept_step
        for _ in range(steps):
            sat = step()
        if reuse_graph:  # only now, so that a call that fails keeps nothing
            self._kept_step = step, optimizer, key
        return float(sat)

    def refute(
        self,
        formula,
        q: float = 0.95,
        steps: int = 2000,
        alpha: float = 0.05,
        beta: float = 10.0,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
    ) -> tuple[float, float]:
        """Reasoning by refutation, as in tutorials/4-reasoning.ipynb: look for a grounding that
        satisfies the axioms (sat >= q) but falsifies `formula`. Returns the final satisfaction
        of the axioms and truth value of `formula`. This is a local search: a counterexample
        shows that `formula` is not a consequence, but a failed search proves nothing. Start
        from fresh groundings, since saturated predicates have vanishing gradients."""
        if isinstance(formula, str):
            formula = self.problem.parse_formula(formula)
        optimizer = optimizer or tf.keras.optimizers.Adam(0.01)
        params = self.trainable_variables

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                sat_kb = self.sat(tape=tape)
                truth = self.evaluate(formula, tape=tape).tensor
                loss = truth + tf.keras.activations.elu(beta * (q - sat_kb), alpha=alpha)
            grads = tape.gradient(loss, params)
            pairs = [(g, p) for g, p in zip(grads, params, strict=True) if g is not None]
            if pairs:
                optimizer.apply_gradients(pairs)
            return sat_kb, truth

        for _ in range(steps):
            sat_kb, truth = step()
        return float(sat_kb), float(truth)

    def _distinct_values(self) -> ltn.core.Formula | None:
        """The implicit axiom that different values are unequal, as one formula: for every pair of
        individuals with different values, they are not equal. None if there are fewer than two."""
        names, seen = [], set()
        for name, value in self.problem.values.items():
            if value not in seen:  # one name for each value, the one that has the grounding
                seen.add(value)
                names.append(name)
        if len(names) < 2:
            return None
        k = len(names)
        values = tf.stack([self.constants[name].tensor for name in names])
        pairs = [tf.repeat(values, k, axis=0), tf.tile(values, [k, 1])]
        equal = tf.reshape(self.predicates["="].model(pairs), [k, k])
        unequal = tf.boolean_mask(1. - equal, tf.logical_not(tf.eye(k, dtype=tf.bool)))
        return ltn.core.Formula(self.quantifiers["!"].aggreg_op(unequal, axis=0), [])

    def _variable(
        self, label: str, individuals: list[ltn.core.Term], tape: tf.GradientTape | None
    ) -> ltn.Variable:
        if not individuals:
            raise ValueError(
                "Variables have nothing to range over, because the problem has no constants. "
                "Pass extra_individuals=1 or more to Grounding, or fill its domain."
            )
        if tape is not None:
            return ltn.Variable.from_constants(label, individuals, tape=tape)
        with warnings.catch_warnings():  # no gradients needed, e.g. when querying
            warnings.simplefilter("ignore")
            return ltn.Variable.from_constants(label, individuals)

    @staticmethod
    def _check(groundings: dict, arities: dict, name: str, arity: int, kind: str) -> None:
        """A clear error for a symbol that a query uses wrongly, instead of a KeyError."""
        if name not in groundings:
            raise ValueError(f"{name} is not a {kind} of the problem.")
        if arities.get(name, arity) != arity:
            expected = arities[name]
            plural = "s" * (expected != 1)
            raise ValueError(f"The {kind} {name} takes {expected} argument{plural}, not {arity}.")

    @staticmethod
    def _apply(symbol: ltn.Predicate | ltn.Function, args: list) -> ltn.core.Expression:
        """LTN takes a single argument as it is, and several arguments as a list."""
        return symbol(args[0] if len(args) == 1 else args)

    def _eval(self, f, env: dict, make_variable):
        recurse = lambda g: self._eval(g, env, make_variable)
        if isinstance(f, Var):
            return env[f.name]
        if isinstance(f, Fn):
            if not f.args:
                self._check(self.constants, {}, f.name, 0, "constant")
                return self.constants[f.name]
            self._check(self.functions, self.problem.functions, f.name, len(f.args), "function")
            return self._apply(self.functions[f.name], [recurse(a) for a in f.args])
        if isinstance(f, Atom):
            if not f.args:
                self._check(self.propositions, {}, f.pred, 0, "proposition")
                return self.propositions[f.pred]
            if f.pred == "=":
                values = [
                    self.problem.values.get(a.name) if isinstance(a, Fn) and not a.args else None
                    for a in f.args
                ]
                if None not in values:  # both are interpreted as themselves
                    return ltn.Proposition(1. if values[0] == values[1] else 0., trainable=False)
            else:
                arity = len(f.args)
                self._check(self.predicates, self.problem.predicates, f.pred, arity, "predicate")
            return self._apply(self.predicates[f.pred], [recurse(a) for a in f.args])
        if isinstance(f, Truth):
            return ltn.Proposition(1. if f.value else 0., trainable=False)
        if isinstance(f, Not):
            return self.Not(recurse(f.arg))
        if isinstance(f, BinOp):
            return self.connectives[f.op](recurse(f.left), recurse(f.right))
        if isinstance(f, Chain):
            return functools.reduce(self.connectives[f.op], [recurse(a) for a in f.args])
        if isinstance(f, Quant):
            bound = {name: make_variable(name) for name in f.vars}
            body = self._eval(f.body, {**env, **bound}, make_variable)
            # LTN rejects a quantifier over a variable that the body doesn't use
            used = [v for v in bound.values() if v.label in body.free_vars]
            return self.quantifiers[f.kind](used, body) if used else body
        raise TypeError(f"Unknown node {f!r}.")
