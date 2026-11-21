"""The parser companion tutorial runs from top to bottom with either Keras generation."""

import json
import pathlib
import unittest

import pytest

pytest.importorskip("lark")


class TestParserTutorial(unittest.TestCase):
    def test_run_all_code_cells(self) -> None:
        path = pathlib.Path(__file__).parents[1] / "tutorials" / "5-parser_and_tptp.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace = {"__name__": "__main__"}
        for index, cell in enumerate(notebook["cells"], start=1):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"{path.name}:cell {index}", "exec"), namespace)
