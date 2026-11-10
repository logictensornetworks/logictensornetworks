# ltn.parser

Reads logic written as text into LTN objects. It needs `lark`, which you get with:

```
pip install ltn[parser]
```

## TPTP

[TPTP](https://tptp.org) is a library of problems for theorem provers, and the text format they are written in. `ltn.parser.tptp` reads its FOF and CNF languages:

```python
from ltn.parser import Grounding, tptp

problem = tptp.parse("""
    fof(socrates, axiom, man(socrates)).
    fof(mortals, axiom, ! [X] : ( man(X) => mortal(X) )).
""")
grounding = Grounding(problem)
grounding.train(500)
print(float(grounding.evaluate("mortal(socrates)").tensor))
```

`Grounding` gives every symbol a default: constants are trainable embeddings, functions and predicates are MLPs, and propositions are trainable truth values. Replace any of them before training, for example with `grounding.predicates["man"] = ...`. Variables range over the constants of the problem, and also over function terms with `Grounding(problem, herbrand_depth=1)`.

Every call to `grounding.train` builds a new TensorFlow graph, which takes a while on a large problem. If you call it repeatedly, for example to log between calls, `grounding.train(10, reuse_graph=True)` keeps the graph and the optimizer. That also changes the results, because the optimizer's state carries on. Use it only if you are sure that the graph hasn't changed: a kept graph sees new values of constants and weights, but it silently ignores a new predicate, connective or domain, and the other things listed in the docstring of `Grounding.train`.

`tptp.parse` raises `SyntaxError` if the text is not valid TPTP, and `NotImplementedError` if it is valid TPTP that the reader doesn't support, such as `include` or the `tff` and `thf` languages.

## Adding a syntax

Each syntax only has to turn text into the syntax tree:

- `tree.py` -- the syntax tree, which all syntaxes share,
- `grounding.py` -- turns a syntax tree into LTN objects, whatever syntax it was read from,
- `tptp.py` and `tptp.lark` -- the TPTP parser and its grammar.

To add a syntax, add a module with a `parse(text)` function that returns a `Problem`, and a function that reads a single formula, which `Grounding` uses for queries given as text. Don't import it from `ltn/parser/__init__.py`, so that only people who use it need its dependencies.
