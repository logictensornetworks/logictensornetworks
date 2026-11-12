"""Reads logic written as text into LTN objects. See README.md in this folder.

`import ltn` doesn't import this package, and this package doesn't import the parsers, so only
the parser you use needs its dependencies.
"""

from ltn.parser.grounding import Grounding
from ltn.parser.tree import Problem
