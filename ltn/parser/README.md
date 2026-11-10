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

`Grounding` gives every symbol a default: constants are trainable embeddings, functions and predicates are MLPs, and propositions are trainable truth values. Replace any of them before training, for example with `grounding.predicates["man"] = ...`. Variables range over the constants of the problem, and also over function terms with `Grounding(problem, herbrand_depth=1)`. If a problem has no constants, give its variables something to range over with `Grounding(problem, extra_individuals=1)` or more.

`grounding.sat()` aggregates the axioms into one truth value, which is 1 for a problem without axioms, and `grounding.train()` maximizes it. A CNF problem usually puts its conjecture in `negated_conjecture` clauses, which are not axioms. To train on every clause, pass `statements=problem.statements`.

TPTP interprets numbers and distinct objects such as `"Apple"` as themselves. An equation between two of them is true or false according to their values, so `"Apple" != "Microsoft"` and `1/2 = 0.5` are true. `sat()` also includes the implicit axiom that different values are unequal, so that training keeps them apart.

LTN evaluates a formula on every combination of the individuals that its variables range over, so a formula with v variables over n individuals builds a tensor with n^v entries. Problems with many constants and variables can run out of memory: about a third of the problems in the TPTP library would need more than a million entries.

Every call to `grounding.train` builds a new TensorFlow graph, which takes a while on a large problem. If you call it repeatedly, for example to log between calls, `grounding.train(10, reuse_graph=True)` keeps the graph and the optimizer. That also changes the results, because the optimizer's state carries on. Use it only if you are sure that the graph hasn't changed: a kept graph sees new values of constants and weights, but it silently ignores a new predicate, connective or domain, and the other things listed in the docstring of `Grounding.train`.

### Files

`tptp.parse_file` reads a file and the files it includes. An include is looked up next to the file that has it, then in the folder given as `root`, which defaults to the environment variable `TPTP`. Problems in the TPTP library include axiom files by names such as `'Axioms/GRP003-0.ax'`, so pass the folder of the library:

```python
problem = tptp.parse_file("TPTP-v9.3.1/Problems/GRP/GRP001-1.p", root="TPTP-v9.3.1")
```

### Errors

`tptp.parse` and `tptp.parse_file` raise `SyntaxError` if the text is not valid TPTP, and `NotImplementedError` if it is valid TPTP that the reader doesn't support, such as the `tff` and `thf` languages, or a formula nested too deeply for Python's recursion limit. `tptp.parse` also refuses `include`, which needs `parse_file`, and `parse_file` raises `FileNotFoundError` if it can't find an included file. A query with a symbol that the problem doesn't have raises `ValueError`.

## Adding a syntax

Each syntax only has to turn text into the syntax tree:

- `tree.py` -- the syntax tree, which all syntaxes share,
- `grounding.py` -- turns a syntax tree into LTN objects, whatever syntax it was read from,
- `tptp.py` and `tptp.lark` -- the TPTP parser and its grammar.

To add a syntax, add a module with a `parse(text)` function that returns a `Problem`, and a function that reads a single formula, which `Grounding` uses for queries given as text. If the syntax has constants that it interprets as themselves, like TPTP's numbers, put their values in `Problem.values`. Don't import it from `ltn/parser/__init__.py`, so that only people who use it need its dependencies.
