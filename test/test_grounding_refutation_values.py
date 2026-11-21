"""Refutation reports the grounding left by its final optimizer update."""

import pytest
import tensorflow as tf

pytest.importorskip("lark")
from ltn.parser import Grounding, tptp


def test_refutation_does_not_report_a_stale_satisfaction_above_the_threshold() -> None:
    grounding = Grounding(tptp.parse("fof(fact, axiom, p)."))
    optimizer = tf.keras.optimizers.SGD(0.1)
    threshold = 0.45
    assert float(grounding.sat()) > threshold

    sat, truth = grounding.refute("p", q=threshold, steps=1, optimizer=optimizer)

    assert sat < threshold
    assert sat == pytest.approx(float(grounding.sat()), abs=1e-6)
    assert truth == pytest.approx(float(grounding.evaluate("p").tensor), abs=1e-6)
    assert int(optimizer.iterations) == 1


@pytest.mark.parametrize("steps", [1, 3])
def test_refutation_reports_the_final_value_of_a_computed_query(steps: int) -> None:
    grounding = Grounding(tptp.parse("fof(fact, axiom, p)."))
    optimizer = tf.keras.optimizers.SGD(0.1)
    query = "~ p"

    result = grounding.refute(query, steps=steps, optimizer=optimizer)

    expected = float(grounding.sat()), float(grounding.evaluate(query).tensor)
    assert result == pytest.approx(expected, abs=1e-6)
    assert int(optimizer.iterations) == steps
