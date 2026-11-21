"""Equal interpreted values must remain one individual after replacing groundings."""

import pathlib
import tempfile
import unittest

import pytest
import tensorflow as tf

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp
from ltn.parser.tree import Atom, Fn, Problem, Statement


def _grounding() -> Grounding:
    grounding = Grounding(tptp.parse("fof(fact, axiom, p(1) & ~ p(1.0))."), embedding_size=1)
    grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
    return grounding


class TestInterpretedValueAliases(unittest.TestCase):
    def test_queries_resolve_alternate_spellings_of_newly_added_values(self) -> None:
        for name, aliases in (
            ("2", ("2.0", "4/2", "2E0")),
            ("0.5", ("1/2", "+2/4", "5E-1")),
            ("-1", ("-1.0", "-2/2")),
            ("0", ("-0.0", "+0/1")),
        ):
            with self.subTest(name=name):
                grounding = Grounding(tptp.parse("fof(fact, axiom, p(f(a)))."), embedding_size=1)
                grounding.constants[name] = ltn.Constant([0.25], trainable=False)
                grounding.functions["f"] = ltn.Function.Lambda(lambda value: value + 0.125)
                grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
                functions, values = (
                    dict(grounding.problem.functions),
                    dict(grounding.problem.values),
                )
                constants, domain = dict(grounding.constants), list(grounding.domain)
                parameters = [id(p) for p in grounding.trainable_variables]

                for alias in aliases:
                    with self.subTest(alias=alias):
                        query = f"p(f({alias}))"
                        for formula in (query, tptp.parse_formula(query)):
                            self.assertEqual(float(grounding.evaluate(formula).tensor), 0.375)
                        self.assertEqual(
                            float(tf.function(lambda: grounding.evaluate(query).tensor)()), 0.375
                        )
                for term in ("3", f"'{name}'", f'"{name}"'):
                    with (
                        self.subTest(ungrounded=term),
                        self.assertRaisesRegex(ValueError, "not a constant"),
                    ):
                        grounding.evaluate(f"p({term})")
                self.assertEqual(grounding.problem.functions, functions)
                self.assertEqual(grounding.problem.values, values)
                self.assertEqual(grounding.constants, constants)
                self.assertEqual(grounding.domain, domain)
                self.assertEqual([id(p) for p in grounding.trainable_variables], parameters)

    def test_added_value_query_aliases_support_gradients_training_and_refutation(self) -> None:
        results = []
        for query in ("p(2)", "p(2.0)"):
            grounding = Grounding(tptp.parse("fof(fact, axiom, p(a))."), embedding_size=1)
            grounding.constants["a"] = ltn.Constant([1.0], trainable=False)
            grounding.constants["2"] = ltn.Constant([0.25], trainable=True)
            parameter = grounding.constants["2"].tensor
            grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: tf.sigmoid(value[:, 0]))
            with tf.GradientTape() as tape:
                truth = grounding.evaluate(query, tape=tape).tensor
            gradient = tape.gradient(truth, parameter)
            self.assertIsNotNone(gradient)

            chosen = tptp.parse(f"fof(chosen, axiom, {query}).").statements
            optimizer = tf.keras.optimizers.SGD(0.05)
            grounding.train(1, optimizer, reuse_graph=True, statements=chosen)
            grounding.train(1, reuse_graph=True, statements=chosen)
            trained = float(parameter[0])
            self.assertGreater(trained, 0.25)
            self.assertEqual(int(optimizer.iterations), 2)
            sat, truth = grounding.refute(query, steps=2, optimizer=tf.keras.optimizers.SGD(0.05))
            self.assertLess(float(parameter[0]), trained)
            self.assertEqual(grounding.domain, ["a"])
            self.assertEqual(grounding.problem.values, {})
            results.append((float(gradient[0]), trained, sat, truth, float(parameter[0])))
        for actual, expected in zip(*results, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_added_alias_cannot_split_an_existing_value(self) -> None:
        for alias in ("1.0", "+1", "2/2", "1E0"):
            with self.subTest(alias=alias):
                grounding = Grounding(tptp.parse("fof(fact, axiom, p(1))."), embedding_size=1)
                grounding.constants["1"].tensor.assign([1.0])
                grounding.constants[alias] = ltn.Constant([0.0], trainable=True)
                grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
                operations = {
                    "evaluate": lambda: grounding.evaluate(f"p(1) & ~ p({alias})"),
                    "explicit individuals": lambda: grounding.evaluate("p(1)", individuals=[]),
                    "sat": grounding.sat,
                    "train": lambda: grounding.train(1),
                    "zero-step train": lambda: grounding.train(0),
                    "refute": lambda: grounding.refute("p(1)", steps=1),
                    "zero-step refute": lambda: grounding.refute("p(1)", steps=0),
                    "parameters": lambda: grounding.trainable_variables,
                    "individuals": grounding.individuals,
                }
                for operation, call in operations.items():
                    with (
                        self.subTest(operation=operation),
                        self.assertRaisesRegex(ValueError, "same interpreted value"),
                    ):
                        call()
                self.assertEqual(float(grounding.constants["1"].tensor[0]), 1.0)
                self.assertEqual(float(grounding.constants[alias].tensor[0]), 0.0)
                self.assertEqual(set(grounding.problem.values), {"1"})
                self.assertEqual(grounding.problem.functions, {"1": 0})
                self.assertEqual(grounding.domain, ["1"])

    def test_added_aliases_share_parameters_without_changing_the_problem(self) -> None:
        grounding = Grounding(tptp.parse("fof(fact, axiom, p(1))."), embedding_size=1)
        shared = grounding.constants["1"]
        shared.tensor.assign([0.25])
        grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
        for alias in ("1.0", "+1", "2/2", "1E0"):
            grounding.constants[alias] = shared
        functions, values = dict(grounding.problem.functions), dict(grounding.problem.values)
        constants, domain = dict(grounding.constants), list(grounding.domain)

        for query in ("p(1.0)", "p(2/2)", "p(3/3)", "! [X] : p(X)"):
            self.assertAlmostEqual(float(grounding.evaluate(query).tensor), 0.25, places=4)
        self.assertEqual([id(p) for p in grounding.trainable_variables], [id(shared.tensor)])
        self.assertEqual(grounding.individuals(), [shared])
        self.assertEqual(grounding.problem.functions, functions)
        self.assertEqual(grounding.problem.values, values)
        self.assertEqual(grounding.constants, constants)
        self.assertEqual(grounding.domain, domain)

    def test_added_aliases_are_checked_before_reusing_a_training_graph(self) -> None:
        for add_first in (False, True):
            with self.subTest(add_first=add_first):
                grounding = Grounding(tptp.parse("fof(fact, axiom, p(1))."), embedding_size=1)
                grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
                if add_first:
                    grounding.constants["1.0"] = grounding.constants["1"]
                optimizer = tf.keras.optimizers.SGD(0.0)
                grounding.train(1, optimizer, reuse_graph=True)
                grounding.constants["1.0"] = ltn.Constant([0.0], trainable=True)

                with self.assertRaisesRegex(ValueError, "same interpreted value"):
                    grounding.train(1, reuse_graph=True)
                self.assertEqual(int(optimizer.iterations), 1)

    def test_extra_values_are_checked_without_interpreting_custom_names(self) -> None:
        grounding = Grounding(tptp.parse("fof(fact, axiom, p(1))."), embedding_size=1)
        grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
        for name in (
            "",
            "custom",
            "01",
            "1_0",
            "1bad",
            "+a",
            "1/0",
            "1 / 2",
            " 1.0 ",
            '"unclosed',
            "'1'",
            '"1"',
        ):
            grounding.constants[name] = ltn.Constant([0.75], trainable=False)
            self.assertEqual(float(grounding.evaluate(Atom("p", (Fn(name),))).tensor), 0.75)
        self.assertEqual(float(grounding.evaluate("p('1')").tensor), 0.75)
        self.assertEqual(float(grounding.evaluate('p("1")').tensor), 0.75)

        # Even two newly grounded spellings of a value absent from the problem must agree.
        grounding.constants["2"] = ltn.Constant([0.25], trainable=False)
        grounding.constants["2.0"] = ltn.Constant([0.75], trainable=False)
        with self.assertRaisesRegex(ValueError, "same interpreted value"):
            grounding.evaluate("p(2)")

    def test_file_problems_also_validate_added_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "problem.p"
            path.write_text("fof(fact, axiom, p(1)).", encoding="utf-8")
            grounding = Grounding(tptp.parse_file(path))
        grounding.constants["1.0"] = ltn.Constant([0.0], trainable=False)
        with self.assertRaisesRegex(ValueError, "same interpreted value"):
            grounding.individuals()

    def test_other_syntaxes_can_interpret_added_constant_names(self) -> None:
        problem = Problem(
            [Statement("fact", "axiom", Atom("p", (Fn("one", value=1),)))],
            lambda text: Atom("p", (Fn(text),)),
            interpret_constant=lambda name: 1 if name == "uno" else None,
        )
        grounding = Grounding(problem, embedding_size=1)
        grounding.constants["uno"] = ltn.Constant([0.0], trainable=False)
        with self.assertRaisesRegex(ValueError, "same interpreted value"):
            grounding.evaluate("uno")

    def test_explicit_problem_values_work_without_a_name_interpreter(self) -> None:
        problem = Problem(
            [],
            lambda name: Atom("p", (Fn(name),)),
            predicates={"p": 1},
            functions={"one": 0, "uno": 0},
            values={"one": 1, "uno": 1},
        )
        grounding = Grounding(problem, embedding_size=1)
        self.assertIs(grounding.constants["one"], grounding.constants["uno"])
        grounding.constants["uno"] = ltn.Constant([0.0], trainable=False)
        with self.assertRaisesRegex(ValueError, "same interpreted value"):
            grounding.evaluate("uno")

    def test_queries_reuse_numeric_values_without_adding_constants(self) -> None:
        for name, aliases in (
            ("1", ("1.0", "+1", "2/2", "1E0")),
            ("1/2", ("0.5", "+2/4", "5E-1")),
            ("-1", ("-1.0", "-2/2")),
            ("0", ("-0.0", "+0/1")),
        ):
            with self.subTest(name=name):
                problem = tptp.parse(f"fof(fact, axiom, p(f({name}))).")
                grounding = Grounding(problem, embedding_size=1)
                grounding.constants[name].tensor.assign([0.25])
                grounding.functions["f"] = ltn.Function.Lambda(lambda value: value + 0.125)
                grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
                functions, values = dict(problem.functions), dict(problem.values)
                constants, domain = dict(grounding.constants), list(grounding.domain)
                parameters = [id(p) for p in grounding.trainable_variables]

                for alias in aliases:
                    with self.subTest(alias=alias):
                        query = f"p(f({alias}))"
                        for formula in (query, tptp.parse_formula(query)):
                            self.assertEqual(float(grounding.evaluate(formula).tensor), 0.375)
                        self.assertEqual(
                            float(tf.function(lambda: grounding.evaluate(query).tensor)()), 0.375
                        )

                self.assertEqual(problem.functions, functions)
                self.assertEqual(problem.values, values)
                self.assertEqual(grounding.constants, constants)
                self.assertEqual(grounding.domain, domain)
                self.assertEqual([id(p) for p in grounding.trainable_variables], parameters)

                grounding.constants[name] = ltn.Constant([0.75], trainable=False)
                self.assertEqual(float(grounding.evaluate(f"p({aliases[0]})").tensor), 0.75)

    def test_query_aliases_work_in_quantifiers_and_mixed_equalities(self) -> None:
        grounding = Grounding(tptp.parse("fof(fact, axiom, 1 = a)."), embedding_size=1)
        grounding.constants["1"].tensor.assign([0.25])
        grounding.constants["a"].tensor.assign([0.75])
        for query, reference in (
            ("2/2 = a", "1 = a"),
            ("a != 1.0", "a != 1"),
            ("! [X] : X = 1E0", "! [X] : X = 1"),
        ):
            with self.subTest(query=query):
                expected = float(grounding.evaluate(reference).tensor)
                self.assertAlmostEqual(float(grounding.evaluate(query).tensor), expected)
                self.assertAlmostEqual(
                    float(tf.function(lambda: grounding.evaluate(query).tensor)()), expected
                )

    def test_query_aliases_preserve_gradients_and_refutation_updates(self) -> None:
        results = []
        for query in ("p(1)", "p(2/2)"):
            grounding = Grounding(tptp.parse("fof(fact, axiom, p(1))."), embedding_size=1)
            parameter = grounding.constants["1"].tensor
            parameter.assign([0.25])
            grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: tf.sigmoid(value[:, 0]))
            with tf.GradientTape() as tape:
                truth = grounding.evaluate(query, tape=tape).tensor
            gradient = tape.gradient(truth, parameter)
            self.assertIsNotNone(gradient)
            sat, truth = grounding.refute(query, steps=2, optimizer=tf.keras.optimizers.SGD(0.05))
            results.append((float(gradient[0]), sat, truth, float(parameter[0])))
        for actual, expected in zip(*results, strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_query_aliases_do_not_ground_other_values_or_quoted_names(self) -> None:
        grounding = Grounding(tptp.parse("fof(fact, axiom, p(1))."))
        for term in ("2", "0.5", "'1'", '"1"'):
            with self.subTest(term=term), self.assertRaisesRegex(ValueError, "not a constant"):
                grounding.evaluate(f"p({term})")

    def test_replacing_one_alias_cannot_satisfy_a_contradiction(self) -> None:
        for name in ("1", "1.0"):
            with self.subTest(replaced=name):
                grounding = _grounding()
                # Either replacement would make p(1) true and p(1.0) false if accepted.
                grounding.constants["1"].tensor.assign([0.0 if name == "1" else 1.0])
                self.assertLess(float(grounding.sat()), 0.01)
                grounding.constants[name] = ltn.Constant(
                    [1.0 if name == "1" else 0.0], trainable=True
                )
                operations = {
                    "evaluate": lambda: grounding.evaluate("p(1) & ~ p(1.0)"),
                    "explicit individuals": lambda: grounding.evaluate("p(1)", individuals=[]),
                    "sat": grounding.sat,
                    "train": lambda: grounding.train(1),
                    "parameters": lambda: grounding.trainable_variables,
                }
                for operation, call in operations.items():
                    with self.subTest(operation=operation), self.assertRaises(ValueError):
                        call()

                replacement = ltn.Constant([0.25], trainable=False)
                grounding.constants["1"] = grounding.constants["1.0"] = replacement
                self.assertEqual(float(grounding.evaluate("p(1)").tensor), 0.25)
                self.assertEqual(float(grounding.evaluate("p(1.0)").tensor), 0.25)
                self.assertLess(float(grounding.sat()), 0.26)
                self.assertEqual(grounding.trainable_variables, [])

    def test_assign_and_training_keep_both_aliases_on_one_parameter(self) -> None:
        grounding = _grounding()
        shared = grounding.constants["1"]
        shared.tensor.assign([0.25])

        self.assertIs(grounding.constants["1.0"], shared)
        self.assertEqual(float(grounding.evaluate("p(1.0)").tensor), 0.25)
        self.assertEqual([id(p) for p in grounding.trainable_variables], [id(shared.tensor)])
        grounding.train(1)

        self.assertGreater(float(shared.tensor[0]), 0.25)
        self.assertEqual(
            float(grounding.evaluate("p(1)").tensor),
            float(grounding.evaluate("p(1.0)").tensor),
        )

    def test_missing_alias_keeps_the_normal_missing_grounding_error(self) -> None:
        grounding = _grounding()
        grounding.constants["1"].tensor.assign([0.25])
        del grounding.constants["1.0"]

        self.assertEqual(float(grounding.evaluate("p(1)").tensor), 0.25)
        with self.assertRaisesRegex(ValueError, "has no grounding"):
            grounding.evaluate("p(1.0)")
        with self.assertRaisesRegex(ValueError, "not a constant"):
            grounding.evaluate("p(2)")

    def test_kept_training_graph_does_not_hide_split_aliases(self) -> None:
        grounding = _grounding()
        grounding.constants["1"].tensor.assign([0.25])
        grounding.train(1, reuse_graph=True)
        grounding.constants["1.0"] = ltn.Constant([0.0], trainable=True)

        with self.assertRaisesRegex(ValueError, "same interpreted value"):
            grounding.train(1, reuse_graph=True)
