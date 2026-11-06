"""Turns a syntax tree into LTN objects. It doesn't depend on the syntax the tree was read from.

Formulas are parsed once into a syntax tree. Every call to `Theory.sat()` evaluates the tree again,
so the LTN variables built from trainable constants are created while the caller's GradientTape is
recording.
"""
from __future__ import annotations

import functools
import itertools
import warnings

import tensorflow as tf

import ltn
from ltn.parser.tree import AXIOM_ROLES, Atom, BinOp, Chain, Fn, Not, Problem, Quant, Statement, Truth, Var


def smooth_equality(alpha: float = 1.0) -> ltn.Predicate:
    """exp(-alpha * squared distance). Unlike exp(-alpha * distance), its gradient is
    defined when both arguments are equal."""
    return ltn.Predicate.Lambda(
        lambda args: tf.exp(-alpha * tf.reduce_sum(tf.square(args[0] - args[1]), axis=1)))


class Theory:
    """Default groundings for the symbols of a problem: constants are trainable embeddings,
    functions and predicates are MLPs, propositions are trainable truth values. Variables
    range over the constants in `domain` and, with `herbrand_depth` > 0, over the functions
    applied to them up to that depth. Every grounding can be replaced before training."""

    def __init__(self, problem: Problem, embedding_size: int = 8, extra_individuals: int = 0,
                 herbrand_depth: int = 0) -> None:
        self.problem = problem
        self.herbrand_depth = herbrand_depth
        d = embedding_size
        names = [f for f, arity in problem.functions.items() if arity == 0]
        names += [f"_individual{i}" for i in range(extra_individuals)]
        self.constants = {c: ltn.Constant(tf.random.uniform([d]), trainable=True) for c in names}
        self.functions = {f: ltn.Function.MLP([d] * n, d) for f, n in problem.functions.items() if n > 0}
        self.predicates = {p: ltn.Predicate.MLP([d] * n) for p, n in problem.predicates.items() if n > 0}
        self.predicates["="] = smooth_equality()
        self.propositions = {p: ltn.Proposition(0.5, trainable=True)
                             for p, n in problem.predicates.items() if n == 0}
        self.domain = list(self.constants)
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
        return params

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
        return self._eval(formula, {}, lambda name: self._variable(name, individuals, tape))

    def sat(self, statements: list[Statement] | None = None, tape: tf.GradientTape | None = None) -> tf.Tensor:
        if statements is None:
            statements = self.problem.with_role(*AXIOM_ROLES)
        individuals = self.individuals()
        return self.aggregator([self.evaluate(s.formula, tape, individuals) for s in statements]).tensor

    def train(self, steps: int, optimizer: tf.keras.optimizers.Optimizer | None = None) -> float:
        """Maximizes the satisfaction of the axioms. Returns the last satisfaction."""
        optimizer = optimizer or tf.keras.optimizers.Adam(0.01)
        params = self.trainable_variables

        @tf.function
        def step():
            with tf.GradientTape() as tape:
                sat = self.sat(tape=tape)
                loss = 1. - sat
            grads = tape.gradient(loss, params)
            optimizer.apply_gradients([(g, p) for g, p in zip(grads, params) if g is not None])
            return sat

        for _ in range(steps):
            sat = step()
        return float(sat)

    def refute(self, formula, q: float = 0.95, steps: int = 2000, alpha: float = 0.05, beta: float = 10.,
               optimizer: tf.keras.optimizers.Optimizer | None = None) -> tuple[float, float]:
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
            optimizer.apply_gradients([(g, p) for g, p in zip(grads, params) if g is not None])
            return sat_kb, truth

        for _ in range(steps):
            sat_kb, truth = step()
        return float(sat_kb), float(truth)

    def _variable(self, label: str, individuals: list[ltn.core.Term], tape: tf.GradientTape | None) -> ltn.Variable:
        if tape is not None:
            return ltn.Variable.from_constants(label, individuals, tape=tape)
        with warnings.catch_warnings():  # no gradients needed, e.g. when querying
            warnings.simplefilter("ignore")
            return ltn.Variable.from_constants(label, individuals)

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
                return self.constants[f.name]
            return self._apply(self.functions[f.name], [recurse(a) for a in f.args])
        if isinstance(f, Atom):
            if not f.args:
                return self.propositions[f.pred]
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
            used = [v for v in bound.values() if v.label in body.free_vars]  # LTN rejects vacuous ones
            return self.quantifiers[f.kind](used, body) if used else body
        raise TypeError(f"Unknown node {f!r}.")
