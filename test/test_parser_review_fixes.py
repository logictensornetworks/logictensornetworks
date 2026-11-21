"""Regressions retained when reconciling the stacked parser review fixes."""

import re
import unittest

import pytest

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp
from ltn.parser.tree import Atom, Fn, Not, Problem, Refused, Statement, Var


class TestParserReviewFixes(unittest.TestCase):
    def test_truth_constants_reject_arguments_clearly(self) -> None:
        for word in ("$true", "$false"):
            for language in ("fof", "cnf", "query"):
                with self.subTest(word=word, language=language):
                    formula = f"{word}(a, b)"
                    text = formula if language == "query" else f"{language}(t, axiom, {formula})."
                    parse = tptp.parse_formula if language == "query" else tptp.parse
                    with self.assertRaisesRegex(
                        SyntaxError, f"{re.escape(word)} at line 1 takes no arguments"
                    ):
                        parse(text)

    def test_unbound_variables_in_formula_trees_report_their_names(self) -> None:
        grounding = Grounding(tptp.parse("fof(fact, axiom, p(f(a)))."))
        for formula, name in (
            (Atom("p", (Var("X"),)), "X"),
            (Atom("p", (Fn("f", (Var("Y"),)),)), "Y"),
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, f"Free variable {name} is not bound"),
            ):
                grounding.evaluate(formula)

    def test_distinct_objects_store_decoded_character_values(self) -> None:
        problem = tptp.parse(
            r'fof(fact, axiom, p("a\"b") & p("c\\d") & p("Apple") & p("1") & p(1)).'
        )
        expected = {
            r'"a\"b"': ("distinct object", 'a"b'),
            r'"c\\d"': ("distinct object", "c\\d"),
            '"Apple"': ("distinct object", "Apple"),
            '"1"': ("distinct object", "1"),
            "1": ("number", 1),
        }
        self.assertEqual(problem.values, expected)
        grounding = Grounding(problem)
        for query, truth in (
            (r'"a\"b" = "a\"b"', 1.0),
            (r'"a\"b" != "c\\d"', 1.0),
            ('1 = "1"', 0.0),
        ):
            with self.subTest(query=query):
                self.assertEqual(float(grounding.evaluate(query).tensor), truth)
        self.assertEqual(problem.values, expected)

    def test_refused_nodes_preserve_the_original_error(self) -> None:
        refused = Refused(NotImplementedError("unsupported construct"))
        for formula in (refused, Not(refused), Atom("p", (refused,))):
            with (
                self.subTest(formula=formula),
                self.assertRaisesRegex(NotImplementedError, "unsupported construct"),
            ):
                Problem([Statement("fact", "axiom", formula)], tptp.parse_formula)
