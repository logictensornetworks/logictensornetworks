# ltn.parser

Reads logic written as text into LTN objects. It needs `lark`, which you get with:

```
pip install ltn[parser]
```

## TPTP

[TPTP](https://tptp.org) is a library of problems for theorem provers, and the text format they are written in. `ltn.parser.tptp` reads its FOF and CNF languages:

```python
from ltn.parser import Theory, tptp

problem = tptp.parse("""
    fof(socrates, axiom, man(socrates)).
    fof(mortals, axiom, ! [X] : ( man(X) => mortal(X) )).
""")
theory = Theory(problem)
theory.train(500)
print(float(theory.evaluate("mortal(socrates)").tensor))
```

`Theory` gives every symbol a default grounding: constants are trainable embeddings, functions and predicates are MLPs, and propositions are trainable truth values. Replace any of them before training, for example with `theory.predicates["man"] = ...`. Variables range over the constants of the problem, and also over function terms with `Theory(problem, herbrand_depth=1)`.

`tptp.parse` raises `SyntaxError` if the text is not valid TPTP, and `NotImplementedError` if it is valid TPTP that the reader doesn't support, such as `include` or the `tff` and `thf` languages.

## Adding a syntax

Each syntax only has to turn text into the syntax tree:

- `tree.py` -- the syntax tree, which all syntaxes share,
- `theory.py` -- turns a syntax tree into LTN objects, whatever syntax it was read from,
- `tptp.py` and `tptp.lark` -- the TPTP parser and its grammar.

To add a syntax, add a module with a `parse(text)` function that returns a `Problem`, and a function that reads a single formula, which `Theory` uses for queries given as text. Don't import it from `ltn/parser/__init__.py`, so that only people who use it need its dependencies.
