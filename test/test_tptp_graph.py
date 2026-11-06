"""Tests that the reader builds the right LTN graph (logictensornetworks#26).

test_tptp.py compares truth values, which shows what a formula computes under one grounding. It
cannot show how the formula is wired: a constant can be cut off from the gradients and every truth
value stays the same. So this file looks at what the reader builds, in four ways:

1. Graph: the TensorFlow graph of each program is, operation by operation, the graph of its LTN core
   definition.
2. Gradients: inside a GradientTape, each program has the same gradients as its LTN core definition.
3. Groundings: a problem gets one LTN object per symbol, of the right kind and size.
4. Training: training a problem read from TPTP gives the same parameters as training the same problem
   written by hand in LTN. Reasoning by refutation is compared in the same way.

Tests 1 and 2 use the tables of test_tptp.py, so they add nothing new to review. The only new thing
to review is the pair PROBLEM and by_hand(), under "A whole problem".
"""
import hashlib

import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Theory, tptp

from . import test_tptp as tables

# ------------------------------------------------------------------------------- Trainable grounding

CONSTANTS = {"a": "a", "b": "b", "c": "c", "'not a word'": "word", "123": "number",
             "'123'": "quoted_number", '"a"': "distinct"}


@pytest.fixture
def parameters(monkeypatch) -> list[tf.Variable]:
    """Makes the constants and propositions of test_tptp.py trainable, for one test. The tables get
    the new objects too, because their `lambda:` looks them up when it runs."""
    variables = []
    for name, attribute in CONSTANTS.items():
        constant = ltn.Constant(getattr(tables, attribute).tensor, trainable=True)
        monkeypatch.setattr(tables, attribute, constant)
        monkeypatch.setitem(tables.CONSTANTS, name, constant)
        variables.append(constant.tensor)
    for name, attribute in (("s", "S"), ("t", "T")):
        proposition = ltn.Proposition(float(getattr(tables, attribute).tensor), trainable=True)
        monkeypatch.setattr(tables, attribute, proposition)
        monkeypatch.setitem(tables.PROPOSITIONS, name, proposition)
        variables.append(proposition.tensor)
    return variables


def read(program: str, **options) -> Theory:
    """The reader's interpretation of `program`. The default equality is `exp(-1. * d)`, and Eq is
    `exp(-d)`: they have the same value, which test_tptp.py checks, but not the same operations."""
    theory = tables.read(program, **options)
    theory.predicates["="] = tables.Eq
    return theory


def reader_column(theory: Theory, tape: tf.GradientTape) -> list[tf.Tensor]:
    return [theory.evaluate(statement.formula, tape=tape).tensor for statement in theory.problem.statements]


def core_column(core, program: str, tape: tf.GradientTape, monkeypatch) -> list[tf.Tensor]:
    """The right column of a table, with x, y and z built from the constants inside the tape, as the
    reader does. They have the labels of the TPTP variables because LTN iterates over sets of labels,
    so the order of some axes depends on how a label is spelled."""
    labels = ("P", "Q", "Z") if "friends" in program else ("X", "Y", "Z")
    for attribute, label in zip("xyz", labels):
        individuals = [tables.a, tables.b, tables.c]
        monkeypatch.setattr(tables, attribute, ltn.Variable.from_constants(label, individuals, tape=tape))
    formulas = core()
    return [formula.tensor for formula in (formulas if isinstance(formulas, list) else [formulas])]


# --------------------------------------------------------------------------------------- Comparing

def fingerprint(graph: tf.Graph) -> list[str]:
    """One hash per output of `graph`. It depends on the operations that compute the output and on
    how they are connected, not on their names or on the order they were created in. Constants
    count by value and variables by identity."""
    captured = {internal.name: external for external, internal in graph.captures}
    hashes = {}

    def visit(tensor):
        if tensor.name not in hashes:
            external = captured.get(tensor.name)
            if external is not None and external.dtype == tf.resource:
                parts = "variable", id(external)
            elif (value := external.numpy() if external is not None else tf.get_static_value(tensor)) is not None:
                parts = "value", value.dtype.name, value.shape, value.tobytes()
            else:
                attributes = sorted((key, attribute.SerializeToString(deterministic=True))
                                    for key, attribute in tensor.op.node_def.attr.items() if not key.startswith("_"))
                parts = tensor.op.type, attributes, tensor.value_index, [visit(i) for i in tensor.op.inputs]
            hashes[tensor.name] = hashlib.sha256(repr(parts).encode()).hexdigest()
        return hashes[tensor.name]

    return [visit(output) for output in graph.outputs]


def graph_of(build) -> list[str]:
    """The fingerprint of the graph of `build(tape)`, which returns a list of tensors."""
    def traced():
        with tf.GradientTape() as tape:
            return build(tape)
    return fingerprint(tf.function(traced).get_concrete_function().graph)


def assert_same_gradients(first, second, variables: list[tf.Variable]) -> None:
    """`first` and `second` are `build` functions, as in graph_of(). A missing gradient only matches
    a missing gradient, so a part of the graph that is cut off fails the test."""
    with tf.GradientTape(persistent=True) as tape:
        first_outputs = first(tape)
        second_outputs = second(tape)
    assert len(first_outputs) == len(second_outputs)
    for output, other in zip(first_outputs, second_outputs):
        for gradient, expected in zip(tape.gradient(output, variables), tape.gradient(other, variables)):
            if expected is None:
                assert gradient is None
            else:
                assert gradient is not None
                assert gradient.numpy() == pytest.approx(expected.numpy(), abs=1e-5)


def test_fingerprint_tells_graphs_apart(parameters):
    a, b, P, Q, R, And, Or = (getattr(tables, name) for name in ("a", "b", "P", "Q", "R", "And", "Or"))
    graph = lambda build: graph_of(lambda tape: [build().tensor])
    assert graph(lambda: P(a)) == graph(lambda: P(a))
    assert graph(lambda: P(a)) != graph(lambda: P(b))                            # another constant
    assert graph(lambda: P(a)) != graph(lambda: Q(a))                            # another predicate
    assert graph(lambda: R([a, b])) != graph(lambda: R([b, a]))                  # another order
    assert graph(lambda: And(P(a), Q(b))) != graph(lambda: Or(P(a), Q(b)))       # another connective


# 1. Graph

@pytest.mark.parametrize("name", tables.PROGRAMS)
def test_graph_matches_core_definition(name, parameters, monkeypatch):
    program, core = tables.PROGRAMS[name]
    theory = read(program)
    assert graph_of(lambda tape: reader_column(theory, tape)) == graph_of(lambda tape: core_column(core, program, tape, monkeypatch))


def test_graph_with_function_terms(parameters):
    a, b, c, F, P, Forall = (getattr(tables, name) for name in ("a", "b", "c", "F", "P", "Forall"))

    def core(tape):
        u = ltn.Variable.from_constants("X", [a, b, c, F(a), F(b), F(c)], tape=tape)
        return [Forall(u, P(F(u))).tensor]

    theory = read("fof(test, axiom, ! [X] : p(f(X))).", herbrand_depth=1)
    assert graph_of(lambda tape: reader_column(theory, tape)) == graph_of(core)


# 2. Gradients

@pytest.mark.parametrize("name", tables.PROGRAMS)
def test_gradients_match_core_definition(name, parameters, monkeypatch):
    program, core = tables.PROGRAMS[name]
    theory = read(program)
    assert_same_gradients(lambda tape: reader_column(theory, tape),
                          lambda tape: core_column(core, program, tape, monkeypatch), parameters)


def test_gradients_with_function_terms(parameters):
    a, b, c, F, P, Forall = (getattr(tables, name) for name in ("a", "b", "c", "F", "P", "Forall"))

    def core(tape):
        u = ltn.Variable.from_constants("X", [a, b, c, F(a), F(b), F(c)], tape=tape)
        return [Forall(u, P(F(u))).tensor]

    theory = read("fof(test, axiom, ! [X] : p(f(X))).", herbrand_depth=1)
    assert_same_gradients(lambda tape: reader_column(theory, tape), core, parameters)


def test_variables_carry_gradients_to_the_constants(parameters):
    theory = read("fof(test, axiom, ! [X] : p(X)).")
    with tf.GradientTape() as tape:
        truth = theory.evaluate(theory.problem.statements[0].formula, tape=tape).tensor
    a, b, c = tape.gradient(truth, parameters[:3])
    assert a is not None and b is not None and c is not None
    assert all(tf.reduce_any(gradient != 0.) for gradient in (a, b, c))


# ------------------------------------------------------------------------------------- A whole problem
# Left: a problem. Right: the same problem with LTN core objects, starting from the same parameters.
# It has every kind of symbol. Some are used in several statements, which must share their parameters,
# and some have the same arity, which must not.

PROBLEM = """
    fof(fact, axiom, p(a)).
    fof(rule, axiom, ! [X] : ( p(X) => q(f(X), b) )).
    fof(implication, axiom, s => p(b)).
    fof(inequality, axiom, a != g(a, b)).
    fof(other_symbols, axiom, w(h(b))).
    cnf(clause, axiom, ~ q(X, X) | t).
    fof(query, conjecture, ? [X] : q(X, c)).
"""
QUERY = "q(a, b)"
SIZE = 4
STEPS = 25


def by_hand(theory: Theory):
    """Returns the parameters of PROBLEM, in the order of theory.trainable_variables, a function that
    computes the satisfaction of its axioms, and a function that computes the query QUERY."""
    a, b, c = [ltn.Constant(theory.constants[name].tensor, trainable=True) for name in "abc"]
    s, t = [ltn.Proposition(float(theory.propositions[name].tensor), trainable=True) for name in "st"]
    f, h = ltn.Function.MLP([SIZE], SIZE), ltn.Function.MLP([SIZE], SIZE)
    g = ltn.Function.MLP([SIZE, SIZE], SIZE)
    p, w = ltn.Predicate.MLP([SIZE]), ltn.Predicate.MLP([SIZE])
    q = ltn.Predicate.MLP([SIZE, SIZE])
    for symbol, name in ((f, "f"), (g, "g"), (h, "h")):
        symbol.model.set_weights(theory.functions[name].model.get_weights())
    for symbol, name in ((p, "p"), (q, "q"), (w, "w")):
        symbol.model.set_weights(theory.predicates[name].model.get_weights())
    Not, Or, Implies, Forall, Eq = tables.Not, tables.Or, tables.Implies, tables.Forall, tables.Eq

    def sat(tape):
        x = ltn.Variable.from_constants("X", [a, b, c], tape=tape)
        return tables.SatAgg([
            p(a),
            Forall(x, Implies(p(x), q([f(x), b]))),
            Implies(s, p(b)),
            Not(Eq([a, g([a, b])])),
            w(h(b)),
            Forall(x, Or(Not(q([x, x])), t)),
        ]).tensor

    weights = [weight for symbol in (f, g, h, p, q, w) for weight in symbol.model.trainable_variables]
    return [a.tensor, b.tensor, c.tensor, s.tensor, t.tensor, *weights], sat, lambda: q([a, b]).tensor


# 3. Groundings

def test_each_symbol_gets_one_grounding():
    theory = Theory(tptp.parse(PROBLEM), embedding_size=SIZE)
    assert sorted(theory.constants) == ["a", "b", "c"]
    assert sorted(theory.propositions) == ["s", "t"]
    assert sorted(theory.functions) == ["f", "g", "h"]
    assert sorted(theory.predicates) == ["=", "p", "q", "w"]
    assert all(isinstance(constant, ltn.Constant) for constant in theory.constants.values())
    assert all(isinstance(proposition, ltn.Proposition) for proposition in theory.propositions.values())
    assert all(isinstance(function, ltn.Function) for function in theory.functions.values())
    assert all(isinstance(predicate, ltn.Predicate) for predicate in theory.predicates.values())


def test_groundings_have_the_right_size():
    theory = Theory(tptp.parse(PROBLEM), embedding_size=SIZE)
    assert all(constant.tensor.shape == (SIZE,) for constant in theory.constants.values())
    assert [len(symbol.model.inputs) for symbol in (theory.functions["f"], theory.functions["g"])] == [1, 2]
    assert [len(symbol.model.inputs) for symbol in (theory.predicates["p"], theory.predicates["q"])] == [1, 2]
    a, b = theory.constants["a"], theory.constants["b"]
    assert theory.functions["g"]([a, b]).tensor.shape == (SIZE,)
    assert theory.predicates["q"]([a, b]).tensor.shape == ()


def test_all_parameters_are_trainable_variables():
    theory = Theory(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected, _, _ = by_hand(theory)
    parameters = theory.trainable_variables
    assert all(parameter.trainable for parameter in parameters)
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert [parameter.shape for parameter in parameters] == [parameter.shape for parameter in expected]


def test_extra_individuals_are_trainable_and_in_the_domain():
    theory = Theory(tptp.parse(PROBLEM), embedding_size=SIZE, extra_individuals=2)
    assert theory.domain == ["a", "b", "c", "_individual0", "_individual1"]
    assert len(theory.individuals()) == 5
    parameters = {id(parameter) for parameter in theory.trainable_variables}
    assert all(id(constant.tensor) in parameters for constant in theory.constants.values())


def test_equality_has_no_parameters():
    theory = Theory(tptp.parse("fof(test, axiom, a = b)."))
    assert len(theory.trainable_variables) == 2


# 4. Training

def assert_same_parameters(theory, expected_parameters, before):
    for parameter, expected, start in zip(theory.trainable_variables, expected_parameters, before, strict=True):
        assert parameter.numpy() == pytest.approx(expected.numpy(), abs=1e-3)
        assert (parameter.numpy() != start).any()  # every parameter moved


def test_first_gradient_matches_the_hand_written_program():
    theory = Theory(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected_parameters, expected_sat, _ = by_hand(theory)
    with tf.GradientTape(persistent=True) as tape:
        sat = theory.sat(tape=tape)
        other = expected_sat(tape)
    assert float(sat) == pytest.approx(float(other), abs=1e-5)
    gradients = tape.gradient(sat, theory.trainable_variables)
    expected = tape.gradient(other, expected_parameters)
    assert all(gradient is not None for gradient in gradients)
    for gradient, other_gradient in zip(gradients, expected, strict=True):
        assert gradient.numpy() == pytest.approx(other_gradient.numpy(), abs=1e-4)


def test_training_matches_the_hand_written_program():
    theory = Theory(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected_parameters, expected_sat, _ = by_hand(theory)
    optimizer = tf.keras.optimizers.Adam(0.01)
    before = [parameter.numpy().copy() for parameter in theory.trainable_variables]

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            sat = expected_sat(tape)
            loss = 1. - sat
        gradients = tape.gradient(loss, expected_parameters)
        optimizer.apply_gradients(zip(gradients, expected_parameters))
        return sat

    for _ in range(STEPS):
        sat = step()

    assert theory.train(STEPS) == pytest.approx(float(sat), abs=1e-4)
    assert_same_parameters(theory, expected_parameters, before)


def test_refutation_matches_the_hand_written_program():
    theory = Theory(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected_parameters, expected_sat, query = by_hand(theory)
    optimizer = tf.keras.optimizers.Adam(0.01)
    before = [parameter.numpy().copy() for parameter in theory.trainable_variables]

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            sat = expected_sat(tape)
            truth = query()
            loss = truth + tf.keras.activations.elu(10. * (0.95 - sat), alpha=0.05)
        gradients = tape.gradient(loss, expected_parameters)
        optimizer.apply_gradients(zip(gradients, expected_parameters))
        return sat, truth

    for _ in range(STEPS):
        sat, truth = step()

    assert theory.refute(QUERY, steps=STEPS) == pytest.approx((float(sat), float(truth)), abs=1e-4)
    assert_same_parameters(theory, expected_parameters, before)


def test_propositions_stay_truth_values_during_training():
    theory = Theory(tptp.parse("fof(test, axiom, s)."))
    theory.train(200)
    assert 0. <= float(theory.propositions["s"].tensor) <= 1.
