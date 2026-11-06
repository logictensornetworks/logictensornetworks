"""Tests for the TPTP reader (logictensornetworks#26).

There are two kinds of tests:

1. Syntax: every program in this file parses, except those in NOT_TPTP and NOT_SUPPORTED, which
   must be rejected.
2. Values: every program in FORMULAS, CLAUSES and FILES has the same truth value as its equivalent
   written by hand with LTN core objects, under the fixed grounding defined below.

To review the tests, read those three tables line by line. The left column is TPTP. The right
column is LTN core code that never goes through the reader. If you agree that the two columns of
a line mean the same thing, then a passing test shows that the reader gives the program that
meaning.

What counts as valid TPTP comes from the official grammar, SyntaxBNF v9.3.1.6, in
https://github.com/TPTPWorld/SyntaxBNF. The tests only call `tptp.parse`, `tptp.parse_formula` and
`Theory`, so they keep working if the parser is replaced. test_tptp_graph.py uses the same tables to
test the LTN graph that the reader builds.
"""
import functools
import subprocess
import sys

import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Theory, tptp

# ------------------------------------------------------------------------------- The grounding
# Both columns of the tables use these objects, the left column through the reader and the right
# column directly. A TPTP name in uppercase is its LTN object (p is P, f is F, s is S), constants
# keep their name, and the TPTP variables X, Y, Z are x, y, z.
#
# The numbers don't matter, because equivalent programs agree under any grounding. They are only
# chosen so that different programs get different truth values.

# Individuals. Variables range over these three.
points = [[0.1, 0.8], [0.6, 0.3], [0.9, 0.5]]
a, b, c = [ltn.Constant(point, trainable=False) for point in points]
x, y, z = [ltn.Variable(label, points) for label in ("x", "y", "z")]

# Constants that only appear in the lines about names.
word = ltn.Constant([0.2, 0.4], trainable=False)           # 'not a word'
number = ltn.Constant([0.5, 0.7], trainable=False)         # 123
quoted_number = ltn.Constant([0.7, 0.9], trainable=False)  # '123'
distinct = ltn.Constant([0.8, 0.1], trainable=False)       # "a"

# Propositions and truth constants.
S = ltn.Proposition(0.7, trainable=False)
T = ltn.Proposition(0.2, trainable=False)
TRUE = ltn.Proposition(1., trainable=False)
FALSE = ltn.Proposition(0., trainable=False)

# Predicates and functions. They are hand-written rather than trained, so nothing here is random,
# and each one depends on the order of its arguments.
P = ltn.Predicate.Lambda(lambda u: tf.sigmoid(3. * u[:, 0] - 2. * u[:, 1]))
Q = ltn.Predicate.Lambda(lambda u: tf.sigmoid(2. * u[:, 1] - u[:, 0]))
R = ltn.Predicate.Lambda(lambda args: tf.sigmoid(3. * args[0][:, 0] - 2. * args[1][:, 1]))
W = ltn.Predicate.Lambda(lambda args: tf.sigmoid(args[0][:, 0] + args[1][:, 1] - 2. * args[2][:, 0]))
F = ltn.Function.Lambda(lambda u: tf.stack([1. - u[:, 1], u[:, 0]], axis=1))
G = ltn.Function.Lambda(lambda args: args[0] - 0.5 * args[1])

# Equality, connectives, quantifiers and the aggregation of axioms. These are the reader's default
# operators. They are written out here so that changing a default in the reader fails the tests.
Eq = ltn.Predicate.Lambda(lambda args: tf.exp(-tf.reduce_sum(tf.square(args[0] - args[1]), axis=1)))
Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
Equiv = ltn.Wrapper_Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.And_Prod(), ltn.fuzzy_ops.Implies_Reichenbach()))
Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall")
Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=6), semantics="exists")
SatAgg = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError(p=2))

CONSTANTS = {"a": a, "b": b, "c": c, "'not a word'": word, "123": number, "'123'": quoted_number, '"a"': distinct}
PROPOSITIONS = {"s": S, "t": T}
PREDICATES = {"p": P, "q": Q, "r": R, "w": W, "smokes": P, "cancer": Q, "friends": R}
FUNCTIONS = {"f": F, "g": G}


def read(program: str, **options) -> Theory:
    """The reader's interpretation of `program` under the grounding above. Only equality and the
    operators keep the reader's defaults."""
    theory = Theory(tptp.parse(program), **options)
    theory.constants = CONSTANTS
    theory.domain = ["a", "b", "c"]
    theory.propositions = PROPOSITIONS
    theory.functions = {name: FUNCTIONS[name] for name in theory.functions}
    theory.predicates.update({name: PREDICATES[name] for name in theory.predicates if name != "="})
    return theory


# ------------------------------------------------------------------------------ The programs
# Left: a formula, tested as fof(test, axiom, <formula>).
# Right: the same formula with LTN core objects. `lambda:` only delays it until the test runs.

FORMULAS = [
    # Atoms
    ("s",                                          lambda: S),
    ("p(a)",                                       lambda: P(a)),
    ("r(a, b)",                                    lambda: R([a, b])),
    ("r(b, a)",                                    lambda: R([b, a])),
    ("w(a, b, c)",                                 lambda: W([a, b, c])),
    ("$true",                                      lambda: TRUE),
    ("$false",                                     lambda: FALSE),
    ("$false | $true",                             lambda: Or(FALSE, TRUE)),
    ("p(a) & $true",                               lambda: And(P(a), TRUE)),
    ("( p(a) )",                                   lambda: P(a)),
    ("( ( p(a) ) )",                               lambda: P(a)),

    # Terms
    ("p(f(a))",                                    lambda: P(F(a))),
    ("p(f(f(a)))",                                 lambda: P(F(F(a)))),
    ("p(g(a, b))",                                 lambda: P(G([a, b]))),
    ("p(g(b, a))",                                 lambda: P(G([b, a]))),
    ("r(f(a), g(b, f(c)))",                        lambda: R([F(a), G([b, F(c)])])),

    # Equality. The grammar says that t1 != t2 is equivalent to ~ t1 = t2.
    ("a = b",                                      lambda: Eq([a, b])),
    ("a != b",                                     lambda: Not(Eq([a, b]))),
    ("~ a = b",                                    lambda: Not(Eq([a, b]))),
    ("f(a) = g(b, c)",                             lambda: Eq([F(a), G([b, c])])),

    # Connectives, each on its own
    ("~ p(a)",                                     lambda: Not(P(a))),
    ("~ ~ p(a)",                                   lambda: Not(Not(P(a)))),
    ("p(a) & q(b)",                                lambda: And(P(a), Q(b))),
    ("p(a) | q(b)",                                lambda: Or(P(a), Q(b))),
    ("p(a) => q(b)",                               lambda: Implies(P(a), Q(b))),
    ("p(a) <= q(b)",                               lambda: Implies(Q(b), P(a))),
    ("p(a) <=> q(b)",                              lambda: Equiv(P(a), Q(b))),
    ("p(a) <~> q(b)",                              lambda: Not(Equiv(P(a), Q(b)))),
    ("p(a) ~| q(b)",                               lambda: Not(Or(P(a), Q(b)))),
    ("p(a) ~& q(b)",                               lambda: Not(And(P(a), Q(b)))),

    # Grouping. Only & and | can be chained, and they group to the left.
    ("p(a) & q(b) & s",                            lambda: And(And(P(a), Q(b)), S)),
    ("p(a) | q(b) | s",                            lambda: Or(Or(P(a), Q(b)), S)),
    ("p(a) & q(b) & s & t",                        lambda: And(And(And(P(a), Q(b)), S), T)),
    ("( p(a) & q(b) ) | s",                        lambda: Or(And(P(a), Q(b)), S)),
    ("p(a) & ( q(b) | s )",                        lambda: And(P(a), Or(Q(b), S))),
    ("( p(a) & q(b) ) => s",                       lambda: Implies(And(P(a), Q(b)), S)),
    ("p(a) => ( q(b) => s )",                      lambda: Implies(P(a), Implies(Q(b), S))),
    ("( p(a) => q(b) ) => s",                      lambda: Implies(Implies(P(a), Q(b)), S)),
    ("( p(a) <=> q(b) ) <=> s",                    lambda: Equiv(Equiv(P(a), Q(b)), S)),

    # ~ applies to the next unit only
    ("~ p(a) & q(b)",                              lambda: And(Not(P(a)), Q(b))),
    ("~ ( p(a) & q(b) )",                          lambda: Not(And(P(a), Q(b)))),
    ("~ p(a) => ~ q(b)",                           lambda: Implies(Not(P(a)), Not(Q(b)))),

    # Quantifiers
    ("! [X] : p(X)",                               lambda: Forall(x, P(x))),
    ("? [X] : p(X)",                               lambda: Exists(x, P(x))),
    ("! [X] : r(X, a)",                            lambda: Forall(x, R([x, a]))),
    ("! [X] : r(a, X)",                            lambda: Forall(x, R([a, x]))),
    ("! [X] : r(X, X)",                            lambda: Forall(x, R([x, x]))),
    ("! [X,Y] : r(X, Y)",                          lambda: Forall([x, y], R([x, y]))),
    ("? [X,Y] : r(X, Y)",                          lambda: Exists([x, y], R([x, y]))),
    ("! [Y,X] : r(X, Y)",                          lambda: Forall([y, x], R([x, y]))),
    ("! [X,Y,Z] : w(X, Y, Z)",                     lambda: Forall([x, y, z], W([x, y, z]))),
    ("! [X] : p(f(X))",                            lambda: Forall(x, P(F(x)))),
    ("! [X,Y] : p(g(Y, X))",                       lambda: Forall([x, y], P(G([y, x])))),
    ("! [X] : X = a",                              lambda: Forall(x, Eq([x, a]))),
    ("! [X] : X != a",                             lambda: Forall(x, Not(Eq([x, a])))),
    ("! [X] : ( X = a | X != b )",                 lambda: Forall(x, Or(Eq([x, a]), Not(Eq([x, b]))))),
    ("! [X] : ( p(X) & s )",                       lambda: Forall(x, And(P(x), S))),
    ("! [X] : ( p(X) => q(X) )",                   lambda: Forall(x, Implies(P(x), Q(x)))),
    ("? [X] : ( p(X) & q(X) )",                    lambda: Exists(x, And(P(x), Q(x)))),

    # The order of nested quantifiers matters
    ("! [X] : ! [Y] : r(X, Y)",                    lambda: Forall(x, Forall(y, R([x, y])))),
    ("! [X] : ? [Y] : r(X, Y)",                    lambda: Forall(x, Exists(y, R([x, y])))),
    ("? [Y] : ! [X] : r(X, Y)",                    lambda: Exists(y, Forall(x, R([x, y])))),
    ("! [X] : ( p(X) => ? [Y] : r(X, Y) )",        lambda: Forall(x, Implies(P(x), Exists(y, R([x, y]))))),

    # A quantifier applies to the next unit only: the q(a) below is outside its scope
    ("! [X] : p(X) & q(a)",                        lambda: And(Forall(x, P(x)), Q(a))),
    ("? [X] : p(X) | q(a)",                        lambda: Or(Exists(x, P(x)), Q(a))),
    ("! [X] : ( p(X) & q(a) )",                    lambda: Forall(x, And(P(x), Q(a)))),
    ("! [X] : ~ p(X) & q(a)",                      lambda: And(Forall(x, Not(P(x))), Q(a))),
    ("~ ! [X] : p(X)",                             lambda: Not(Forall(x, P(x)))),
    ("~ ? [X] : p(X)",                             lambda: Not(Exists(x, P(x)))),
    ("! [X] : ~ p(X)",                             lambda: Forall(x, Not(P(x)))),

    # An inner quantifier hides an outer variable of the same name
    ("! [X] : ( p(X) & ? [X] : q(X) )",            lambda: Forall(x, And(P(x), Exists(y, Q(y))))),
    ("! [X] : ( ? [X] : q(X) & p(X) )",            lambda: Forall(x, And(Exists(y, Q(y)), P(x)))),
    ("! [X] : ? [X] : p(X)",                       lambda: Exists(x, P(x))),
    ("( ! [X] : p(X) ) & ( ? [X] : q(X) )",        lambda: And(Forall(x, P(x)), Exists(x, Q(x)))),

    # A variable that doesn't occur in the body changes nothing
    ("! [X] : p(a)",                               lambda: P(a)),
    ("? [X] : p(a)",                               lambda: P(a)),
    ("! [X,Y] : p(X)",                             lambda: Forall(x, P(x))),
    ("! [X,Y] : ( s & ? [X] : ~ ! [Z] : ~ t )",    lambda: And(S, Not(Not(T)))),

    # Names. 'cat' and cat are the same word, but '123' is not the number 123, and a distinct
    # object such as "a" is different from the constant a.
    ("'p'(a)",                                     lambda: P(a)),
    ("p('a')",                                     lambda: P(a)),
    ("'r'('a', 'b')",                              lambda: R([a, b])),
    ("'s'",                                        lambda: S),
    ("p('not a word')",                            lambda: P(word)),
    ("p(123)",                                     lambda: P(number)),
    ("p('123')",                                   lambda: P(quoted_number)),
    ('p("a")',                                     lambda: P(distinct)),

    # Larger formulas
    ("! [X] : ? [Y] : ( r(X, Y) & ~ r(Y, X) )",
        lambda: Forall(x, Exists(y, And(R([x, y]), Not(R([y, x])))))),
    ("~ ! [X] : ? [Y] : ( r(X, Y) => ~ ? [Z] : ~ ( p(Y) | q(Z) ) )",
        lambda: Not(Forall(x, Exists(y, Implies(R([x, y]), Not(Exists(z, Not(Or(P(y), Q(z)))))))))),
    ("! [X,Y] : ( q(X) => ( ~ p(X) & ( f(X) = Y => X != c ) & r(X, g(Y, f(c))) ) )",
        lambda: Forall([x, y], Implies(Q(x), And(And(
            Not(P(x)),
            Implies(Eq([F(x), y]), Not(Eq([x, c])))),
            R([x, G([y, F(c)])]))))),
]

# Left: a clause, tested as cnf(test, axiom, <clause>).
# Right: the same clause with LTN core objects.

CLAUSES = [
    # Literals
    ("p(a)",                                       lambda: P(a)),
    ("s",                                          lambda: S),
    ("~ p(a)",                                     lambda: Not(P(a))),
    ("~ ( p(a) )",                                 lambda: Not(P(a))),
    ("a = b",                                      lambda: Eq([a, b])),
    ("a != b",                                     lambda: Not(Eq([a, b]))),
    ("~ a = b",                                    lambda: Not(Eq([a, b]))),
    ("$false",                                     lambda: FALSE),

    # Disjunctions
    ("p(a) | q(b)",                                lambda: Or(P(a), Q(b))),
    ("p(a) | q(b) | s",                            lambda: Or(Or(P(a), Q(b)), S)),
    ("~ p(a) | a = b | a != c",                    lambda: Or(Or(Not(P(a)), Eq([a, b])), Not(Eq([a, c])))),
    ("( p(a) | q(b) )",                            lambda: Or(P(a), Q(b))),
    ("( ( p(a) | q(b) ) )",                        lambda: Or(P(a), Q(b))),

    # Variables are implicitly universally quantified over the whole clause
    ("p(X)",                                       lambda: Forall(x, P(x))),
    ("~ p(X) | q(X)",                              lambda: Forall(x, Or(Not(P(x)), Q(x)))),
    ("p(X) | q(Y)",                                lambda: Forall([x, y], Or(P(x), Q(y)))),
    ("r(X, Y) | p(X)",                             lambda: Forall([x, y], Or(R([x, y]), P(x)))),
    ("X = a | X != b",                             lambda: Forall(x, Or(Eq([x, a]), Not(Eq([x, b]))))),
    ("p(f(X)) | r(X, g(X, a))",                    lambda: Forall(x, Or(P(F(x)), R([x, G([x, a])])))),
]


def smokes_friends_cancer():
    """The rules of axioms() in examples/smokes_friends_cancer/smokes_friends_cancer.py, copied
    as they are apart from the p= arguments of the quantifiers, which TPTP cannot express."""
    Smokes, Cancer, Friends = P, Q, R
    p, q = x, y
    axioms = []
    # friendship is anti-reflexive
    axioms.append(Forall(p,Not(Friends([p,p]))))
    # friendship is symmetric
    axioms.append(Forall((p,q),Implies(Friends([p,q]),Friends([q,p]))))
    # everyone has a friend
    axioms.append(Forall(p,Exists(q,Friends([p,q]))))
    # smoking propagates among friends
    axioms.append(Forall((p,q),Implies(And(Friends([p,q]),Smokes(p)),Smokes(q))))
    # smoking causes cancer + not smoking causes not cancer
    axioms.append(Forall(p,Implies(Smokes(p),Cancer(p))))
    axioms.append(Forall(p,Implies(Not(Smokes(p)),Not(Cancer(p)))))
    return axioms


# Left: a name for the test and a whole file.
# Right: one LTN core formula per statement of the file.

FILES = [
    ("comments between tokens", """
        % A line comment.
        /* A block comment
           over two lines. */
        fof(test, axiom, p(a) /* inside a formula */ & % until the end of this line
                         q(b)).
     """,
     lambda: [And(P(a), Q(b))]),

    ("no spaces at all", "fof(test,axiom,![X]:(p(X)=>?[Y]:(r(X,Y)&X!=Y))).",
     lambda: [Forall(x, Implies(P(x), Exists(y, And(R([x, y]), Not(Eq([x, y]))))))]),

    ("no space between statements", "fof(one,axiom,p(a)).cnf(two,axiom,q(b)).",
     lambda: [P(a), Q(b)]),

    ("statement names", """
        fof(1, axiom, p(a)).
        fof('a quoted name', axiom, q(b)).
        fof(name_with_digits_2, axiom, s).
     """,
     lambda: [P(a), Q(b), S]),

    ("annotations are not part of the formula", """
        fof(one, axiom, p(a), file('problem.p', one)).
        fof(two, axiom, q(b), unknown, [useful(1), "text", 12.5, [nested, list]]).
        fof(three, axiom, s, inference(rule, [status(thm)], [one, two]), []).
        fof(four, axiom, t, unknown, [data($fof(! [Y] : ( q(Y) & r(a, Y) )))]).
     """,
     lambda: [P(a), Q(b), S, T]),

    ("fof and cnf in one file", """
        fof(one, axiom, ! [X] : ( p(X) => q(X) )).
        cnf(two, axiom, ~ p(X) | q(X)).
     """,
     lambda: [Forall(x, Implies(P(x), Q(x))), Forall(x, Or(Not(P(x)), Q(x)))]),

    ("clauses do not share variables", """
        cnf(one, axiom, p(X)).
        cnf(two, axiom, q(X)).
     """,
     lambda: [Forall(x, P(x)), Forall(x, Q(x))]),

    ("the rules of examples/smokes_friends_cancer", """
        %----Written by hand from the LTN example. The right column is copied from the example.
        fof(anti_reflexive, axiom, ! [P] : ~ friends(P, P)).
        fof(symmetric, axiom, ! [P,Q] : ( friends(P, Q) => friends(Q, P) )).
        fof(has_a_friend, axiom, ! [P] : ? [Q] : friends(P, Q)).
        fof(smoking_propagates, axiom, ! [P,Q] : ( ( friends(P, Q) & smokes(P) ) => smokes(Q) )).
        fof(smoking_causes_cancer, axiom, ! [P] : ( smokes(P) => cancer(P) )).
        fof(no_smoking_no_cancer, axiom, ! [P] : ( ~ smokes(P) => ~ cancer(P) )).
     """,
     smokes_friends_cancer),
]

# Which statements form the knowledge base. Every role except the two kinds of conjecture counts
# as an axiom. The grammar gives `plain` no meaning, and reading it as an axiom is our choice.

KNOWLEDGE_BASE = """
    fof(role_axiom, axiom, p(a)).
    fof(role_hypothesis, hypothesis, ! [X] : ( p(X) => q(X) )).
    fof(role_definition, definition, ! [X] : ( q(X) <=> r(X, X) )).
    fof(role_assumption, assumption, s).
    fof(role_lemma, lemma, p(b)).
    fof(role_theorem, theorem, q(b)).
    fof(role_corollary, corollary, p(c)).
    fof(role_plain, plain, q(c)).
    cnf(role_clause, axiom, ~ p(X) | q(X)).
    fof(role_conjecture, conjecture, r(a, b)).
    cnf(role_negated_conjecture, negated_conjecture, ~ r(a, b)).
"""


def knowledge_base():
    return SatAgg([
        P(a),
        Forall(x, Implies(P(x), Q(x))),
        Forall(x, Equiv(Q(x), R([x, x]))),
        S,
        P(b),
        Q(b),
        P(c),
        Q(c),
        Forall(x, Or(Not(P(x)), Q(x))),
    ])


# Valid programs that only need to parse.

PARSE_ONLY = [
    # Files without statements
    "",
    "% only a comment",
    "/* only a comment */",

    # Numbers are terms
    "fof(test, axiom, p(0)).",
    "fof(test, axiom, p(-1)).",
    "fof(test, axiom, p(+1)).",
    "fof(test, axiom, p(1/2)).",
    "fof(test, axiom, p(-1/2)).",
    "fof(test, axiom, p(1.5)).",
    "fof(test, axiom, p(-1.5)).",
    "fof(test, axiom, p(1E3)).",
    "fof(test, axiom, p(1.5E3)).",
    "fof(test, axiom, p(1.5e-3)).",
    "fof(test, axiom, p(1.5E+3)).",
    "fof(test, axiom, p(1E-3)).",
    "fof(test, axiom, p(1E+3)).",
    "fof(test, axiom, p(0.0)).",
    "fof(test, axiom, p(0/1)).",
    "fof(test, axiom, p(007E3)).",  # the grammar allows leading zeros before an exponent

    # Quoted words and distinct objects
    r"fof(test, axiom, p('it\'s')).",
    r"fof(test, axiom, p('back\\slash')).",
    r'fof(test, axiom, p("a \"quoted\" object")).',
    "fof(test, axiom, p('% not a comment')).",
    "fof(test, axiom, p('/* not a comment */')).",
    "fof(test, axiom, '$s'(a)).",  # an ordinary predicate, because the $ is quoted

    # Ordinary words that look special
    "fof(test, axiom, equal(a, b)).",
    "fof(fof, axiom, cnf(include)).",
    "fof(axiom, axiom, axiom).",
    "fof(test, axiom, ! [X1,Y_2] : r(X1, Y_2)).",
    "cnf(test, axiom, ( ( ( p(a) ) ) )).",

    # Statement names
    "fof(-1, axiom, p(a)).",
    "fof(+1, axiom, p(a)).",
    "fof(`Name, axiom, p(a)).",

    # Annotations have no effect on the formula, so what they hold is no reason to refuse a program
    "fof(test, axiom, p(a), unknown).",
    "fof(test, axiom, p(a), file('a.p'), [data(name):[colon, list, 2]]).",
    "cnf(test, plain, $false, inference(rule, [status(thm), [X, $fot(X), a]], [1, 2])).",
    "fof(test, axiom, p(a), file('a.p', `Name)).",
    "fof(test, axiom, p(a), file(include, tff)).",
    "fof(test, axiom, p(a), [include, tff, thf, fof, cnf, axiom]).",
    "fof(test, axiom, p(a), unknown, [1, -1, +2, 1/2, 1.5, 1E3, 1.5E-3]).",
    "fof(test, axiom, p(a), unknown, [$fof($less(1, 2))]).",
    "fof(test, axiom, p(a), unknown, [$fof($$system(a))]).",
    "fof(test, axiom, p(a), unknown, [$fof([q(a)] --> [s])]).",
    "fof(test, axiom, p(a), unknown, [$fof(`P(a))]).",
    "fof(test, axiom, p(a), unknown, [$fot($sum(1, 2))]).",
    "fof(test, axiom, p(a), unknown, [$cnf($less(1, 2))]).",
]

# Programs that are not valid TPTP.

NOT_TPTP = [
    # Statements
    ("p(a).",                                      "no language"),
    ("FOF(x, axiom, p(a)).",                       "languages are lowercase"),
    ("fof(x, axiom, p(a))",                        "no final dot"),
    ("fof(x, axiom, p(a).",                        "no closing parenthesis"),
    ("fof(x, p(a)).",                              "no role"),
    ("fof(x, axiom).",                             "no formula"),
    ("fof(x, axiom, ).",                           "empty formula"),
    ("fof(X, axiom, p(a)).",                       "a name cannot be a variable"),
    ("fof(1.5, axiom, p(a)).",                     "a name cannot be a real number"),
    ("fof(x, banana, p(a)).",                      "not a role"),
    ("fof(x, unknown, p(a)).",                     "the grammar calls this role an error"),
    ("fof(x, axiom, p(a)) fof(y, axiom, q(a)).",   "no dot between statements"),
    ("fof(x, axiom, p(a)). garbage",               "text after the last statement"),
    ("fof",                                        "file cut off after the language"),
    ("fof(",                                       "file cut off before the name"),
    ("fof(x, axiom",                               "file cut off after the role"),
    ("fof(x, axiom, ! [X",                         "file cut off in a quantifier"),
    ("fof(x, axiom, p(a",                          "file cut off in a term"),
    ("fof(x, axiom, p(a), file('a.p'",             "file cut off in an annotation"),

    # Connectives cannot be mixed or chained without parentheses
    ("fof(x, axiom, p(a) & q(a) | s).",            "& then |"),
    ("fof(x, axiom, p(a) | q(a) & s).",            "| then &"),
    ("fof(x, axiom, p(a) & q(a) => s).",           "& then =>"),
    ("fof(x, axiom, p(a) => q(a) | s).",           "=> then |"),
    ("fof(x, axiom, p(a) => q(a) => s).",          "=> is not associative"),
    ("fof(x, axiom, p(a) <= q(a) <= s).",          "<= is not associative"),
    ("fof(x, axiom, p(a) <=> q(a) <=> s).",        "<=> is not associative"),
    ("fof(x, axiom, p(a) <~> q(a) <~> s).",        "<~> is not associative"),
    ("fof(x, axiom, p(a) ~| q(a) ~| s).",          "~| is not associative"),
    ("fof(x, axiom, p(a) ~& q(a) ~& s).",          "~& is not associative"),

    # Missing operands
    ("fof(x, axiom, p(a) &).",                     "& without a right operand"),
    ("fof(x, axiom, & p(a)).",                     "& without a left operand"),
    ("fof(x, axiom, ~).",                          "~ without an operand"),

    # Quantifiers
    ("fof(x, axiom, ! [] : p(a)).",                "no variables"),
    ("fof(x, axiom, ! [a] : p(a)).",               "a constant cannot be quantified"),
    ("fof(x, axiom, ! X : p(X)).",                 "no brackets"),
    ("fof(x, axiom, ! [X] p(X)).",                 "no colon"),
    ("fof(x, axiom, ! [X] : ).",                   "no body"),
    ("fof(x, axiom, ! [X : $i] : p(X)).",          "typed variables are TFF"),

    # All variables of a FOF formula must be quantified
    ("fof(x, axiom, p(X)).",                       "free variable"),
    ("fof(x, axiom, ! [X] : r(X, Y)).",            "Y is free"),
    ("fof(x, axiom, ! [X] : p(X) => q(X)).",       "the X of q(X) is outside the quantifier"),

    # Atoms and terms
    ("fof(x, axiom, X).",                          "a variable is not a formula"),
    ("fof(x, axiom, ! [X] : X).",                  "a quantified variable is not a formula"),
    ("fof(x, axiom, X(a)).",                       "a variable is not a predicate"),
    ("fof(x, axiom, p(X(a))).",                    "a variable is not a function"),
    ("fof(x, axiom, p()).",                        "no arguments in the parentheses"),
    ("fof(x, axiom, p(a,)).",                      "missing argument"),
    ("fof(x, axiom, 42).",                         "a number is not a formula"),
    ('fof(x, axiom, "p").',                        "a distinct object is not a formula"),
    ("fof(x, axiom, p(a & b)).",                   "a formula is not a term"),
    ("fof(x, axiom, p($true)).",                   "$true is not a term"),
    ("fof(x, axiom, a = b = c).",                  "= is not associative"),
    ("fof(x, axiom, a =).",                        "= without a right operand"),

    # Clauses can only be disjunctions of literals
    ("cnf(x, axiom, p(a) & q(a)).",                "no & in clauses"),
    ("cnf(x, axiom, p(a) => q(a)).",               "no => in clauses"),
    ("cnf(x, axiom, ! [X] : p(X)).",               "no quantifiers in clauses"),
    ("cnf(x, axiom, ~ ~ p(a)).",                   "only one ~ per literal"),
    ("cnf(x, axiom, ~ ( p(a) | q(a) )).",          "~ applies to an atom only"),
    ("cnf(x, axiom, ~ a != b).",                   "~ cannot apply to !="),
    ("cnf(x, axiom, p(a) | ( q(a) | s )).",        "no nested disjunction"),
    ("cnf(x, axiom, ( p(a) ) | q(a)).",            "no parentheses around one literal of many"),

    # Characters
    ("fof(x, axiom, p(a) # q(a)).",                "unknown character"),
    ("fof(x, axiom, p('unterminated)).",           "unterminated quote"),
    ("/* unterminated comment",                    "unterminated comment"),

    # Annotations: <annotations> ::= ,<source><optional_info>
    ("fof(x, axiom, p(a), ).",                     "an annotation cannot be empty"),
    ("fof(x, axiom, p(a), ()).",                   "parentheses are not a general term"),
    ("fof(x, axiom, p(a), [(]]).",                 "brackets must match"),
    ("fof(x, axiom, p(a), [a b]).",                "no comma in a list"),
    ("fof(x, axiom, p(a), a b).",                  "two sources"),
    ("fof(x, axiom, p(a), unknown, [], []).",      "three annotations"),
    ("fof(x, axiom, p(a), unknown, a).",           "useful information must be a list"),
    ("fof(x, axiom, p(a), [(! [X] : q(X))]).",     "a formula in an annotation needs $fof(...)"),

    # Numbers: <integer>, <rational> and <real>
    ("fof(x, axiom, p(007)).",                     "leading zeros in an integer"),
    ("fof(x, axiom, p(01.5)).",                    "leading zeros in a real"),
    ("fof(x, axiom, p(1/0)).",                     "a rational cannot have denominator 0"),
    ("fof(x, axiom, p(1/2E3)).",                   "a rational has no exponent"),

    # Quoted words: <sq_char> and <do_char>
    ("fof(x, axiom, p('')).",                      "an empty quoted word"),
    ('fof(x, axiom, p("")).',                      "an empty distinct object"),
    ("fof(x, axiom, p('two\nlines')).",            "a newline in a quoted word"),
    ('fof(x, axiom, p("two\nlines")).',            "a newline in a distinct object"),
    (r"fof(x, axiom, p('a\b')).",                  "a backslash only escapes ' and itself"),
    (r'fof(x, axiom, p("a\b")).',                  'a backslash only escapes " and itself'),
    ("fof(x, axiom, p('café')).",             "only visible ASCII characters"),
    ("fof(x, 'axiom', p(a)).",                     "a role is a <lower_word>, not an <atomic_word>"),

    # Wrong, not just unsupported
    ("include",                                    "include without parentheses"),
    ("tff",                                        "tff without parentheses"),
    ("fof(x, axiom, p(-)).",                       "a minus sign without a number"),
    ("fof(x, axiom, p(`)).",                       "a back quote without a word"),
    ("fof(x, axiom, `p(a)).",                      "only an <upper_word> can be back-quoted"),
    ("fof(x, axiom, [p(a)]).",                     "a sequent without an arrow"),
    ("fof(x, axiom, p(a) & [q(a)] --> [s]).",      "a sequent inside a formula"),
    ("fof(x, axiom, ~ [p(a)] --> [q(a)]).",        "a sequent after ~"),
    ("fof(x, axiom, $foo(a).",                     "no closing parenthesis after a defined symbol"),
    ("fof(x, axiom, [p(a)] --> [q(a)]",            "file cut off after a sequent"),
]

# Valid TPTP that the reader refuses, with a clear error, instead of misreading it.

NOT_SUPPORTED = [
    ("include('Axioms/SET001-0.ax').",             "include"),
    ("include('Axioms/SET001-0.ax', [a, b]).",     "include of selected formulas"),
    ("tff(x, axiom, p(a)).",                       "language tff"),
    ("tff(x, type, p: $int > $o).",                "language tff, with characters fof doesn't have"),
    ("thf(x, axiom, p).",                          "language thf"),
    ("thf(x, axiom, p @ a).",                      "language thf, with characters fof doesn't have"),
    ("tcf(x, axiom, p(a)).",                       "language tcf"),
    ("tpi(x, axiom, p(a)).",                       "language tpi"),
    ("fof(x, axiom, [p(a)] --> [q(a)]).",          "sequents"),
    ("fof(x, type, p(a)).",                        "role type"),
    ("fof(x, interpretation, p(a)).",              "role interpretation"),
    ("fof(x, logic, p(a)).",                       "role logic"),
    ("fof(x, axiom-subrole, p(a)).",               "subroles"),
    ("fof(x, axiom, `P(a)).",                      "back-quoted words"),
    ("fof(x, axiom, $distinct(a, b, c)).",         "$distinct"),
    ("fof(x, axiom, $less(1, 2)).",                "defined predicates"),
    ("fof(x, axiom, p($sum(1, 2))).",              "defined functions"),
    ("fof(x, axiom, $$system(a)).",                "system predicates"),
    ("fof(x, axiom, p($$system(a))).",             "system functions"),
    ("fof(x, axiom, ([p(a)] --> [q(a)])).",        "a sequent in parentheses"),
    ("fof(x, axiom, [] --> []).",                  "a sequent without formulas"),
    ("fof(x, axiom, [p(a), q(a)] --> [s, t]).",    "a sequent with several formulas"),
    ("fof(x, axiom-sub(a, 2):[b], p(a)).",         "a subrole that is a general term"),
    ("fof(x, type-a, p(a)).",                      "role type with a subrole"),
    ("fof(x, axiom, p(a) & $foo).",                "a defined constant"),
    ("fof(x, axiom, $foo = a).",                   "a defined constant in an equation"),
    ("fof(x, axiom, ! [X] : $less(X, 1)).",        "a defined predicate on a variable"),
    ("cnf(x, axiom, $less(X, 1)).",                "a defined predicate in a clause"),
    ("cnf(x, axiom, ~ $less(1, 2)).",              "a negated defined predicate"),
    ("fof(x, axiom, `P).",                         "a back-quoted constant"),
    ("fof(x, axiom, p(`A)).",                      "a back-quoted term"),
    ("fof(x, axiom, p(a), unknown, [$tff(p)]).",   "$tff(...) in an annotation, which needs a TFF parser"),
    ("fof(x, axiom, p(a), unknown, [$thf(p)]).",   "$thf(...) in an annotation, which needs a THF parser"),

    # Groundings are looked up by name, so a name can only have one arity
    ("fof(x, axiom, p(a) & p(a, b)).",             "predicate with two arities"),
    ("fof(x, axiom, s & s(a)).",                   "proposition and predicate of the same name"),
    ("fof(x, axiom, p(f(a)) & p(f(a, b))).",       "function with two arities"),
    ("fof(x, axiom, p(a) & p(a(b))).",             "constant and function of the same name"),
]


# ------------------------------------------------------------------------------- The tests

def programs():
    for formula, core in FORMULAS:
        program = f"fof(test, axiom, {formula})."
        yield program, program, core
    for clause, core in CLAUSES:
        program = f"cnf(test, axiom, {clause})."
        yield program, program, core
    yield from FILES


PROGRAMS = {name: (program, core) for name, program, core in programs()}


def values_read_from(program: str, **options) -> list[float]:
    """The truth value that the reader gives to each statement of `program`."""
    theory = read(program, **options)
    return [float(theory.evaluate(statement.formula).tensor) for statement in theory.problem.statements]


def values_of(*formulas):
    """The truth values of LTN core formulas, to compare with up to rounding errors."""
    return pytest.approx([float(formula.tensor) for formula in formulas], abs=1e-6)


# 1. Syntax

@pytest.mark.parametrize("name", PROGRAMS)
def test_syntax_program_parses(name):
    program, _ = PROGRAMS[name]
    tptp.parse(program)


@pytest.mark.parametrize("program", PARSE_ONLY)
def test_syntax_valid_program_parses(program):
    tptp.parse(program)


@pytest.mark.parametrize("program", [program for program, _ in NOT_TPTP],
                         ids=[reason for _, reason in NOT_TPTP])
def test_syntax_invalid_program_is_rejected(program):
    with pytest.raises(SyntaxError):
        tptp.parse(program)


@pytest.mark.parametrize("program", [program for program, _ in NOT_SUPPORTED],
                         ids=[reason for _, reason in NOT_SUPPORTED])
def test_syntax_unsupported_program_is_rejected(program):
    with pytest.raises(NotImplementedError):
        tptp.parse(program)


@pytest.mark.parametrize("query", ["p(X)", "! [X] : r(X, Y)", "p(a) garbage", "p(a) .", "fof(x, axiom, p(a))."])
def test_syntax_invalid_query_is_rejected(query):
    with pytest.raises(SyntaxError):
        tptp.parse_formula(query)


def test_syntax_long_chains_parse():
    tptp.parse("fof(test, axiom, " + " & ".join(["p(a)"] * 10000) + ").")
    tptp.parse("fof(test, axiom, " + " | ".join(["p(a)"] * 10000) + ").")
    tptp.parse("cnf(test, axiom, " + " | ".join(["p(X)"] * 10000) + ").")


# 2. Values

@pytest.mark.parametrize("name", PROGRAMS)
def test_value_matches_core_definition(name):
    program, core = PROGRAMS[name]
    expected = core()
    expected = expected if isinstance(expected, list) else [expected]
    assert values_read_from(program) == values_of(*expected)


def test_value_of_the_knowledge_base():
    assert [float(read(KNOWLEDGE_BASE).sat())] == values_of(knowledge_base())


def test_value_of_a_query():
    theory = read("fof(test, axiom, p(a) & r(a, b)).")
    assert [float(theory.evaluate("p(b) | r(b, a)").tensor)] == values_of(Or(P(b), R([b, a])))


def test_value_of_a_long_chain():
    # $true leaves a conjunction unchanged, so the value depends on the last operand being there.
    program = "fof(test, axiom, " + " & ".join(["$true"] * 2000 + ["p(a)"]) + ")."
    assert values_read_from(program) == values_of(functools.reduce(And, [TRUE] * 2000 + [P(a)]))


# With herbrand_depth=n, variables also range over function terms up to depth n.

def ranging_over(*individuals) -> ltn.Variable:
    return ltn.Variable("u", tf.stack(ltn.as_tensors(individuals)))


def test_value_with_function_terms_of_depth_1():
    u = ranging_over(a, b, c, F(a), F(b), F(c))
    assert values_read_from("fof(test, axiom, ! [X] : p(f(X))).", herbrand_depth=1) == values_of(Forall(u, P(F(u))))


def test_value_with_function_terms_of_depth_2():
    u = ranging_over(a, b, c, F(a), F(b), F(c), F(F(a)), F(F(b)), F(F(c)))
    assert values_read_from("fof(test, axiom, ! [X] : p(f(X))).", herbrand_depth=2) == values_of(Forall(u, P(F(u))))


def test_value_with_a_binary_function_term():
    u = ranging_over(a, b, c, *[G([v, w]) for v in (a, b, c) for w in (a, b, c)])
    assert values_read_from("fof(test, axiom, ! [X] : q(g(X, a))).", herbrand_depth=1) == values_of(Forall(u, Q(G([u, a]))))


# The parser is optional.

def test_ltn_works_without_lark():
    code = "import sys; sys.modules['lark'] = None; import ltn, ltn.parser; print('ltn works'); import ltn.parser.tptp"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "ltn works" in result.stdout
    assert "pip install ltn[parser]" in result.stderr
