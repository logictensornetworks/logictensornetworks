"""Tests that the reader builds the right LTN graph (logictensornetworks#26).

test_tptp.py compares truth values, which shows what a formula computes under one grounding. It
cannot show how the formula is wired: a constant can be cut off from the gradients and every truth
value stays the same. So this file looks at what the reader builds, in four ways:

1. Graph: the TensorFlow graph of each program is, operation by operation, the graph of its LTN core
   definition.
2. Gradients: inside a GradientTape, each program has the same gradients as its LTN core definition.
3. Groundings: a problem gets one LTN object per symbol, of the right kind and size.
4. Training: training a problem read from TPTP gives the same parameters as training the same
   problem written by hand in LTN. Reasoning by refutation is compared in the same way.

Tests 1 and 2 use the tables of test_tptp.py, so they add nothing new to review. The only new thing
to review is the pair PROBLEM and by_hand(), under "A whole problem".

The last section tests `train(reuse_graph=True)`, which keeps the graph between calls.
"""

import hashlib

import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp

from . import test_tptp as tables

# ------------------------------------------------------------------------------ Trainable grounding

CONSTANTS = {
    "a": "a",
    "b": "b",
    "c": "c",
    "'not a word'": "word",
    "123": "number",
    "'123'": "quoted_number",
    '"a"': "distinct",
}


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


def read(program: str, **options) -> Grounding:
    """The reader's interpretation of `program`. The default equality is `exp(-1. * d)`, and Eq is
    `exp(-d)`: they have the same value, which test_tptp.py checks, but not the same operations."""
    grounding = tables.read(program, **options)
    grounding.predicates["="] = tables.Eq
    return grounding


def reader_column(grounding: Grounding, tape: tf.GradientTape) -> list[tf.Tensor]:
    return [
        grounding.evaluate(statement.formula, tape=tape).tensor
        for statement in grounding.problem.statements
    ]


def core_column(core, program: str, tape: tf.GradientTape, monkeypatch) -> list[tf.Tensor]:
    """The right column of a table, with x, y and z built from the constants inside the tape, as
    the reader does. They have the labels of the TPTP variables because LTN iterates over sets of
    labels, so the order of some axes depends on how a label is spelled."""
    labels = ("P", "Q", "Z") if "friends" in program else ("X", "Y", "Z")
    for attribute, label in zip("xyz", labels, strict=True):
        individuals = [tables.a, tables.b, tables.c]
        variable = ltn.Variable.from_constants(label, individuals, tape=tape)
        monkeypatch.setattr(tables, attribute, variable)
    formulas = core()
    return [formula.tensor for formula in (formulas if isinstance(formulas, list) else [formulas])]


# --------------------------------------------------------------------------------------- Comparing


def fingerprint(graph: tf.Graph) -> list[str]:
    """One hash per output of `graph`. It depends on the operations that compute the output and on
    how they are connected, not on their names or on the order they were created in. Constants
    count by value and variables by identity."""
    captured = {internal.name: external for external, internal in graph.captures}
    hashes = {}

    def value_of(tensor, external):
        return external.numpy() if external is not None else tf.get_static_value(tensor)

    def visit(tensor):
        if tensor.name not in hashes:
            external = captured.get(tensor.name)
            if external is not None and external.dtype == tf.resource:
                parts = "variable", id(external)
            elif (value := value_of(tensor, external)) is not None:
                parts = "value", value.dtype.name, value.shape, value.tobytes()
            else:
                attributes = sorted(
                    (key, attribute.SerializeToString(deterministic=True))
                    for key, attribute in tensor.op.node_def.attr.items()
                    if not key.startswith("_")
                )
                inputs = [visit(i) for i in tensor.op.inputs]
                parts = tensor.op.type, attributes, tensor.value_index, inputs
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
    for output, other in zip(first_outputs, second_outputs, strict=True):
        gradients = tape.gradient(output, variables), tape.gradient(other, variables)
        for gradient, expected in zip(*gradients, strict=True):
            if expected is None:
                assert gradient is None
            else:
                assert gradient is not None
                assert gradient.numpy() == pytest.approx(expected.numpy(), abs=1e-5)


def test_fingerprint_tells_graphs_apart(parameters):
    names = ("a", "b", "P", "Q", "R", "And", "Or")
    a, b, P, Q, R, And, Or = (getattr(tables, name) for name in names)
    graph = lambda build: graph_of(lambda tape: [build().tensor])
    assert graph(lambda: P(a)) == graph(lambda: P(a))
    assert graph(lambda: P(a)) != graph(lambda: P(b))  # another constant
    assert graph(lambda: P(a)) != graph(lambda: Q(a))  # another predicate
    assert graph(lambda: R([a, b])) != graph(lambda: R([b, a]))  # another order
    assert graph(lambda: And(P(a), Q(b))) != graph(lambda: Or(P(a), Q(b)))  # another connective


# 1. Graph


@pytest.mark.parametrize("name", tables.PROGRAMS)
def test_graph_matches_core_definition(name, parameters, monkeypatch):
    program, core = tables.PROGRAMS[name]
    grounding = read(program)
    from_reader = graph_of(lambda tape: reader_column(grounding, tape))
    from_core = graph_of(lambda tape: core_column(core, program, tape, monkeypatch))
    assert from_reader == from_core


def test_graph_with_function_terms(parameters):
    a, b, c, F, P, Forall = (getattr(tables, name) for name in ("a", "b", "c", "F", "P", "Forall"))

    def core(tape):
        u = ltn.Variable.from_constants("X", [a, b, c, F(a), F(b), F(c)], tape=tape)
        return [Forall(u, P(F(u))).tensor]

    grounding = read("fof(test, axiom, ! [X] : p(f(X))).", herbrand_depth=1)
    assert graph_of(lambda tape: reader_column(grounding, tape)) == graph_of(core)


# 2. Gradients


@pytest.mark.parametrize("name", tables.PROGRAMS)
def test_gradients_match_core_definition(name, parameters, monkeypatch):
    program, core = tables.PROGRAMS[name]
    grounding = read(program)
    assert_same_gradients(
        lambda tape: reader_column(grounding, tape),
        lambda tape: core_column(core, program, tape, monkeypatch),
        parameters,
    )


def test_gradients_with_function_terms(parameters):
    a, b, c, F, P, Forall = (getattr(tables, name) for name in ("a", "b", "c", "F", "P", "Forall"))

    def core(tape):
        u = ltn.Variable.from_constants("X", [a, b, c, F(a), F(b), F(c)], tape=tape)
        return [Forall(u, P(F(u))).tensor]

    grounding = read("fof(test, axiom, ! [X] : p(f(X))).", herbrand_depth=1)
    assert_same_gradients(lambda tape: reader_column(grounding, tape), core, parameters)


def test_variables_carry_gradients_to_the_constants(parameters):
    grounding = read("fof(test, axiom, ! [X] : p(X)).")
    with tf.GradientTape() as tape:
        truth = grounding.evaluate(grounding.problem.statements[0].formula, tape=tape).tensor
    a, b, c = tape.gradient(truth, parameters[:3])
    assert a is not None and b is not None and c is not None
    assert all(tf.reduce_any(gradient != 0.0) for gradient in (a, b, c))


# ---------------------------------------------------------------------------------- A whole problem
# Left: a problem. Right: the same problem with LTN core objects, starting from the same parameters.
# It has every kind of symbol. Some are used in several statements, which must share their
# parameters, and some have the same arity, which must not.

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


def by_hand(grounding: Grounding):
    """Returns the parameters of PROBLEM, in the order of grounding.trainable_variables, a function
    that computes the satisfaction of its axioms, and a function that computes the query QUERY."""
    a, b, c = [ltn.Constant(grounding.constants[name].tensor, trainable=True) for name in "abc"]
    s, t = [
        ltn.Proposition(float(grounding.propositions[name].tensor), trainable=True) for name in "st"
    ]
    f, h = ltn.Function.MLP([SIZE], SIZE), ltn.Function.MLP([SIZE], SIZE)
    g = ltn.Function.MLP([SIZE, SIZE], SIZE)
    p, w = ltn.Predicate.MLP([SIZE]), ltn.Predicate.MLP([SIZE])
    q = ltn.Predicate.MLP([SIZE, SIZE])
    for symbol, name in ((f, "f"), (g, "g"), (h, "h")):
        symbol.model.set_weights(grounding.functions[name].model.get_weights())
    for symbol, name in ((p, "p"), (q, "q"), (w, "w")):
        symbol.model.set_weights(grounding.predicates[name].model.get_weights())
    Not, Or, Implies, Forall, Eq = tables.Not, tables.Or, tables.Implies, tables.Forall, tables.Eq

    def sat(tape):
        x = ltn.Variable.from_constants("X", [a, b, c], tape=tape)
        return tables.SatAgg(
            [
                p(a),
                Forall(x, Implies(p(x), q([f(x), b]))),
                Implies(s, p(b)),
                Not(Eq([a, g([a, b])])),
                w(h(b)),
                Forall(x, Or(Not(q([x, x])), t)),
            ]
        ).tensor

    symbols = (f, g, h, p, q, w)
    weights = [weight for symbol in symbols for weight in symbol.model.trainable_variables]
    parameters = [a.tensor, b.tensor, c.tensor, s.tensor, t.tensor, *weights]
    return parameters, sat, lambda: q([a, b]).tensor


# 3. Groundings


def test_each_symbol_gets_one_grounding():
    grounding = Grounding(tptp.parse(PROBLEM), embedding_size=SIZE)
    assert sorted(grounding.constants) == ["a", "b", "c"]
    assert sorted(grounding.propositions) == ["s", "t"]
    assert sorted(grounding.functions) == ["f", "g", "h"]
    assert sorted(grounding.predicates) == ["=", "p", "q", "w"]
    assert all(isinstance(constant, ltn.Constant) for constant in grounding.constants.values())
    propositions = grounding.propositions.values()
    assert all(isinstance(proposition, ltn.Proposition) for proposition in propositions)
    assert all(isinstance(function, ltn.Function) for function in grounding.functions.values())
    assert all(isinstance(predicate, ltn.Predicate) for predicate in grounding.predicates.values())


def test_groundings_have_the_right_size():
    grounding = Grounding(tptp.parse(PROBLEM), embedding_size=SIZE)
    assert all(constant.tensor.shape == (SIZE,) for constant in grounding.constants.values())
    functions, predicates = grounding.functions, grounding.predicates
    assert [len(symbol.model.inputs) for symbol in (functions["f"], functions["g"])] == [1, 2]
    assert [len(symbol.model.inputs) for symbol in (predicates["p"], predicates["q"])] == [1, 2]
    a, b = grounding.constants["a"], grounding.constants["b"]
    assert grounding.functions["g"]([a, b]).tensor.shape == (SIZE,)
    assert grounding.predicates["q"]([a, b]).tensor.shape == ()


def test_all_parameters_are_trainable_variables():
    grounding = Grounding(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected, _, _ = by_hand(grounding)
    parameters = grounding.trainable_variables
    assert all(parameter.trainable for parameter in parameters)
    assert len({id(parameter) for parameter in parameters}) == len(parameters)
    assert [parameter.shape for parameter in parameters] == [other.shape for other in expected]


def test_extra_individuals_are_trainable_and_in_the_domain():
    grounding = Grounding(tptp.parse(PROBLEM), embedding_size=SIZE, extra_individuals=2)
    assert grounding.domain == ["a", "b", "c", "_individual0", "_individual1"]
    assert len(grounding.individuals()) == 5
    parameters = {id(parameter) for parameter in grounding.trainable_variables}
    assert all(id(constant.tensor) in parameters for constant in grounding.constants.values())


def test_equality_has_no_parameters():
    grounding = Grounding(tptp.parse("fof(test, axiom, a = b)."))
    assert len(grounding.trainable_variables) == 2


def test_equal_values_share_one_grounding():
    problem = tptp.parse('fof(test, axiom, p(1/2) & p(0.5) & p(2) & p("2")).')
    grounding = Grounding(problem, embedding_size=SIZE)
    assert grounding.constants["1/2"] is grounding.constants["0.5"]
    assert grounding.domain == ["1/2", "2", '"2"']
    parameters = grounding.trainable_variables
    assert len({id(parameter) for parameter in parameters}) == len(parameters)


def test_variables_need_individuals():
    problem = tptp.parse("fof(test, axiom, ! [X] : p(X)).")
    with pytest.raises(ValueError, match="extra_individuals"):
        Grounding(problem).sat()
    assert 0.0 < float(Grounding(problem, extra_individuals=1).sat()) < 1.0


# 4. Training


def assert_same_parameters(grounding, expected_parameters, before):
    parameters = grounding.trainable_variables
    for parameter, expected, start in zip(parameters, expected_parameters, before, strict=True):
        assert parameter.numpy() == pytest.approx(expected.numpy(), abs=1e-3)
        assert (parameter.numpy() != start).any()  # every parameter moved


def test_first_gradient_matches_the_hand_written_program():
    grounding = Grounding(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected_parameters, expected_sat, _ = by_hand(grounding)
    with tf.GradientTape(persistent=True) as tape:
        sat = grounding.sat(tape=tape)
        other = expected_sat(tape)
    assert float(sat) == pytest.approx(float(other), abs=1e-5)
    gradients = tape.gradient(sat, grounding.trainable_variables)
    expected = tape.gradient(other, expected_parameters)
    assert all(gradient is not None for gradient in gradients)
    for gradient, other_gradient in zip(gradients, expected, strict=True):
        assert gradient.numpy() == pytest.approx(other_gradient.numpy(), abs=1e-4)


def test_training_matches_the_hand_written_program():
    grounding = Grounding(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected_parameters, expected_sat, _ = by_hand(grounding)
    optimizer = tf.keras.optimizers.Adam(0.01)
    before = [parameter.numpy().copy() for parameter in grounding.trainable_variables]

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            sat = expected_sat(tape)
            loss = 1.0 - sat
        gradients = tape.gradient(loss, expected_parameters)
        optimizer.apply_gradients(zip(gradients, expected_parameters, strict=True))
        return sat

    for _ in range(STEPS):
        sat = step()

    assert grounding.train(STEPS) == pytest.approx(float(sat), abs=1e-4)
    assert_same_parameters(grounding, expected_parameters, before)


def test_refutation_matches_the_hand_written_program():
    grounding = Grounding(tptp.parse(PROBLEM), embedding_size=SIZE)
    expected_parameters, expected_sat, query = by_hand(grounding)
    optimizer = tf.keras.optimizers.Adam(0.01)
    before = [parameter.numpy().copy() for parameter in grounding.trainable_variables]

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            sat = expected_sat(tape)
            truth = query()
            loss = truth + tf.keras.activations.elu(10.0 * (0.95 - sat), alpha=0.05)
        gradients = tape.gradient(loss, expected_parameters)
        optimizer.apply_gradients(zip(gradients, expected_parameters, strict=True))
        return sat, truth

    for _ in range(STEPS):
        sat, truth = step()

    expected = pytest.approx((float(sat), float(truth)), abs=1e-4)
    assert grounding.refute(QUERY, steps=STEPS) == expected
    assert_same_parameters(grounding, expected_parameters, before)


def test_training_without_axioms_changes_nothing():
    grounding = Grounding(tptp.parse("fof(test, conjecture, p(a))."), embedding_size=SIZE)
    before = [parameter.numpy().copy() for parameter in grounding.trainable_variables]
    assert grounding.train(2) == 1.0
    assert all(
        (parameter.numpy() == start).all()
        for parameter, start in zip(grounding.trainable_variables, before, strict=True)
    )


def test_training_on_chosen_statements_is_training_on_them_as_axioms():
    # A CNF problem usually puts its conjecture in negated_conjecture clauses, which are not axioms.
    chosen = tptp.parse("cnf(fact, negated_conjecture, p(a)).\ncnf(goal, axiom, ~ q(a)).")
    as_axioms = tptp.parse("cnf(fact, axiom, p(a)).\ncnf(goal, negated_conjecture, ~ q(a)).")
    tf.keras.utils.set_random_seed(0)
    first = Grounding(chosen, embedding_size=SIZE)
    tf.keras.utils.set_random_seed(0)
    second = Grounding(as_axioms, embedding_size=SIZE)
    assert same_parameters(first, second)
    first.train(STEPS, statements=chosen.with_role("negated_conjecture"))
    second.train(STEPS)
    assert same_parameters(first, second)


def test_propositions_stay_truth_values_during_training():
    grounding = Grounding(tptp.parse("fof(test, axiom, s)."))
    grounding.train(200)
    assert 0.0 <= float(grounding.propositions["s"].tensor) <= 1.0


# ------------------------------------------------------------------ Keeping the graph between calls

SMALL_PROBLEM = "fof(test, axiom, p(a))."


def seeded_grounding() -> Grounding:
    """The same grounding of PROBLEM every time, so that the results of training can be compared."""
    tf.keras.utils.set_random_seed(0)
    return Grounding(tptp.parse(PROBLEM), embedding_size=SIZE)


def same_parameters(first: Grounding, second: Grounding) -> bool:
    pairs = zip(first.trainable_variables, second.trainable_variables, strict=True)
    return all(mine.numpy() == pytest.approx(other.numpy(), abs=1e-6) for mine, other in pairs)


def count_traces(grounding: Grounding) -> list:
    """Returns a list that grows when `train` builds a graph, the only time a step calls `sat`. How
    much it grows is not defined: TensorFlow traces twice if the optimizer creates its variables."""
    traces = []
    sat = grounding.sat

    def counted_sat(*args, **kwargs):
        traces.append(None)
        return sat(*args, **kwargs)

    grounding.sat = counted_sat
    return traces


def test_default_calls_do_not_share_an_optimizer():
    twice, once = seeded_grounding(), seeded_grounding()
    assert same_parameters(twice, once)
    twice.train(5)
    twice.train(5)
    once.train(10)
    assert not same_parameters(twice, once)


def test_kept_graph_matches_a_single_call():
    twice, once = seeded_grounding(), seeded_grounding()
    twice.train(5, reuse_graph=True)
    twice.train(5, reuse_graph=True)
    once.train(10)
    assert same_parameters(twice, once)


def test_second_flagged_call_does_not_trace_again():
    grounding = Grounding(tptp.parse(SMALL_PROBLEM), embedding_size=SIZE)
    traces = count_traces(grounding)
    optimizer = tf.keras.optimizers.Adam(0.01)
    grounding.train(1, optimizer, reuse_graph=True)
    built = len(traces)
    assert built > 0
    grounding.train(1, reuse_graph=True)
    grounding.train(1, optimizer, reuse_graph=True)
    assert len(traces) == built
    assert int(optimizer.iterations) == 3


def test_default_call_leaves_the_kept_graph_alone():
    grounding = Grounding(tptp.parse(SMALL_PROBLEM), embedding_size=SIZE)
    traces = count_traces(grounding)
    optimizer = tf.keras.optimizers.Adam(0.01)
    grounding.train(1, optimizer, reuse_graph=True)
    kept = len(traces)
    grounding.train(1)
    rebuilt = len(traces)
    assert rebuilt > kept
    assert int(optimizer.iterations) == 1  # it started a new Adam
    grounding.train(1, reuse_graph=True)
    assert len(traces) == rebuilt
    assert int(optimizer.iterations) == 2


def test_different_optimizer_rebuilds_the_graph_and_is_kept():
    grounding = Grounding(tptp.parse(SMALL_PROBLEM), embedding_size=SIZE)
    traces = count_traces(grounding)
    first, second = tf.keras.optimizers.Adam(0.01), tf.keras.optimizers.Adam(0.01)
    grounding.train(1, first, reuse_graph=True)
    built = len(traces)
    grounding.train(2, second, reuse_graph=True)
    rebuilt = len(traces)
    assert rebuilt > built
    grounding.train(1, reuse_graph=True)
    assert len(traces) == rebuilt
    assert (int(first.iterations), int(second.iterations)) == (1, 3)


def test_failed_call_keeps_no_graph():
    grounding = Grounding(tptp.parse(SMALL_PROBLEM), embedding_size=SIZE)
    predicate = grounding.predicates.pop("p")
    with pytest.raises(ValueError):
        grounding.train(1, reuse_graph=True)
    grounding.predicates["p"] = predicate  # a kept step would still train the parameters without it
    before = [weight.numpy().copy() for weight in predicate.model.trainable_variables]
    grounding.train(1, reuse_graph=True)
    assert all(
        (weight.numpy() != start).any()
        for weight, start in zip(predicate.model.trainable_variables, before, strict=True)
    )


def test_kept_graph_sees_an_assign_to_a_constant():
    grounding = Grounding(tptp.parse("fof(test, axiom, a = b)."), embedding_size=SIZE)
    traces = count_traces(grounding)
    a, b = grounding.constants["a"].tensor, grounding.constants["b"].tensor
    a.assign(tf.zeros(SIZE))
    b.assign(tf.ones(SIZE))
    assert grounding.train(1, reuse_graph=True) < 0.1
    built = len(traces)
    a.assign(b)
    assert grounding.train(1, reuse_graph=True) == pytest.approx(1.0, abs=1e-3)
    assert len(traces) == built


def test_kept_graph_does_not_see_a_new_predicate():
    """A limit of the flag, and the reason why it is not the default."""
    grounding = Grounding(tptp.parse(SMALL_PROBLEM), embedding_size=SIZE)
    grounding.train(1, reuse_graph=True)
    old_sat = float(grounding.sat())
    grounding.predicates["p"] = ltn.Predicate.Lambda(lambda x: tf.ones_like(x[:, 0]))
    assert float(grounding.sat()) == pytest.approx(1.0, abs=1e-3)
    assert grounding.train(1, reuse_graph=True) == pytest.approx(old_sat, abs=1e-6)


def test_kept_graph_is_rebuilt_for_other_statements():
    problem = tptp.parse("fof(fact, axiom, p(a)).\nfof(goal, negated_conjecture, ~ p(a)).")
    grounding = Grounding(problem, embedding_size=SIZE)
    traces = count_traces(grounding)
    grounding.train(1, reuse_graph=True)
    built = len(traces)
    grounding.train(1, reuse_graph=True, statements=problem.statements)
    rebuilt = len(traces)
    assert rebuilt > built
    grounding.train(1, reuse_graph=True, statements=problem.statements)
    assert len(traces) == rebuilt
