"""Equal interpreted values must remain one individual after replacing groundings."""

import unittest

import pytest

import ltn

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


def _grounding() -> Grounding:
    grounding = Grounding(tptp.parse("fof(fact, axiom, p(1) & ~ p(1.0))."), embedding_size=1)
    grounding.predicates["p"] = ltn.Predicate.Lambda(lambda value: value[:, 0])
    return grounding


class TestInterpretedValueAliases(unittest.TestCase):
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
