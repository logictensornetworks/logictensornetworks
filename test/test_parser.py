import unittest

from ltn.parser import parse


class TestParseFormula(unittest.TestCase):
    def test_1(self):
        parse.parse_formula("p(a,b)")