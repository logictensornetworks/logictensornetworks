"""Quantification over unused variables does not need a grounding domain."""

import pytest

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


@pytest.mark.parametrize(
    "formula, equivalent",
    [
        ("! [X] : $true", "$true"),
        ("? [X] : $false", "$false"),
        ("! [X,Y] : p", "p"),
        ("! [X] : (p & ? [X] : q)", "p & q"),
        ("? [X] : ! [X] : p", "p"),
    ],
)
def test_unused_quantifiers_evaluate_without_constants(formula: str, equivalent: str) -> None:
    grounding = Grounding(tptp.parse(f"fof(test, axiom, {formula})."))
    assert grounding.domain == []
    result = grounding.evaluate(formula)
    assert result.free_vars == []
    assert float(result.tensor) == pytest.approx(float(grounding.evaluate(equivalent).tensor))


def test_quantified_proposition_trains_without_constants() -> None:
    grounding = Grounding(tptp.parse("fof(test, axiom, ! [X] : (? [X] : p))."))
    before = float(grounding.propositions["p"].tensor)
    grounding.train(2)
    assert float(grounding.propositions["p"].tensor) > before
    assert float(grounding.evaluate("! [X] : p").tensor) == pytest.approx(
        float(grounding.propositions["p"].tensor)
    )


def test_inner_quantifier_that_uses_its_variable_still_needs_a_domain() -> None:
    formula = "! [X] : (? [X] : p(X))"
    problem = tptp.parse(f"fof(test, axiom, {formula}).")
    with pytest.raises(ValueError, match="extra_individuals"):
        Grounding(problem).sat()
    grounding = Grounding(problem, extra_individuals=2)
    assert float(grounding.evaluate(formula).tensor) == pytest.approx(
        float(grounding.evaluate("? [X] : p(X)").tensor)
    )
