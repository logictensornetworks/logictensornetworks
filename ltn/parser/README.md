# ltn.parser

Reads logic written as text into LTN objects. It needs `lark`, which you get with:

```
pip install ltn[parser]
```

## TPTP

[TPTP](https://tptp.org) is a library of problems for theorem provers, and the text format they are written in. `ltn.parser.tptp` reads its FOF and CNF languages. Evaluation uses fuzzy LTN semantics over a finite grounding domain; high satisfaction is not a classical theorem-proving certificate.

The [companion tutorial](../../tutorials/5-parser_and_tptp.ipynb) compares native and parsed formulas using the same custom groundings, then demonstrates training, queries and refutation.

For a small problem:

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

This prints a truth value close to 1, for example `0.9999988079071045`. The groundings start out random, so the value changes a little from run to run.

`Grounding` gives every symbol a default: constants are trainable embeddings, functions and predicates are MLPs, and propositions are trainable truth values. Replace any of them before training, for example with `grounding.predicates["man"] = ...`. Variables range over the constants of the problem, and also over function terms with `Grounding(problem, herbrand_depth=1)`. If a problem has no constants, give its variables something to range over with `Grounding(problem, extra_individuals=1)` or more.

Every quantified variable uses the same domain, including constants occurring only in conjectures. This differs from native LTN variables with separate domains, such as samples and labels. Textual formulas have no direct equivalent of `ltn.diag`, guarded quantification with `mask=`, or per-call quantifier `p=` and `stable=` overrides. The shared operator objects can be replaced before training. Queries must be closed formulas: use `! [X] : p(X)` rather than `p(X)`.

To add a new function when using Herbrand expansion, set both its grounding in `grounding.functions` and its arity in `problem.functions`. Replacing the grounding of an existing function uses its declared arity.

Herbrand terms are evaluated only when a quantified variable needs the domain. Ground formulas and empty axiom selections do not call functions merely to construct unused individuals.

`grounding.train()` calls models in training mode, so custom Dropout and BatchNormalization layers behave as they do during Keras training. Evaluation, `sat()`, and refutation use inference mode; pass `training=True` to `evaluate()` or `sat()` when building your own training loop.

For a custom training loop, `grounding.trainable_variables` collects the trainable constants, propositions and model weights once each. Fixed constants and propositions are excluded.

`grounding.sat()` aggregates the explicit axioms and the implicit distinctness constraint described below into one truth value, and `grounding.train()` maximizes it. It returns 1 when neither contributes a constraint. A CNF problem usually puts its conjecture in `negated_conjecture` clauses, which are not explicit axioms. To train on every clause, pass `statements=problem.statements`.

With the default stable aggregator, an explicit `$true` axiom has satisfaction about `0.9999`; an unconstrained problem returns exactly `1.0`.

`grounding.refute(formula, q=0.95)` searches for a grounding whose axiom satisfaction is at least `q` while the formula's truth value is below `q`. It changes the grounding and returns the final axiom satisfaction and formula truth value. Start from a fresh grounding: saturated models can have poor search gradients. A counterexample refutes the consequence for the configured LTN problem and threshold; an unsuccessful local search proves nothing.

Ordinary equality, such as `a = b`, defaults to smooth similarity, `exp(-squared_distance(a, b))`, rather than a crisp identity test.

TPTP interprets numbers and distinct objects such as `"Apple"` as themselves. An equation between two of them is true or false according to their values, so `"Apple" != "Microsoft"` and `1/2 = 0.5` are true. These comparisons also work in queries when the literals do not occur in the problem, without adding individuals to the domain. Other uses of a literal, such as `p(2)`, still need a grounding for it.

In `Problem.values`, distinct objects store their character strings with quotes removed and escaped quotes and backslashes decoded. Constant names retain their original TPTP spelling, and distinct objects remain separate from numeric values.

The default embeddings of numeric literals encode their identities, not their magnitudes or numeric features: `p(2)` does not automatically receive `[2.0]`. Replace the constants and models when grounding actual numeric data. Defined arithmetic symbols such as `$less` and `$sum` are unsupported.

Queries reuse existing numeric groundings by value: if the problem contains `p(1)`, queries such as `p(1.0)` and `p(2/2)` use the same `Constant`, without adding names or individuals to the problem. A query such as `p(2)` still fails unless that value has a grounding.

Equal values such as `1` and `1.0` share one `Constant`. To change their grounding, assign to the shared tensor or replace every alias with the same `Constant` object. Replacing only one alias with a different object raises `ValueError` when the grounding is used. This check also covers aliases added directly to `grounding.constants` and runs before cached training and refutation. Adding an alias does not add it to `problem.values` or the quantifier domain.

`sat()` includes the implicit axiom that different interpreted values in the problem are unequal, so that training keeps them apart. This applies even when those values occur only in conjectures, so a problem without explicit axioms can have satisfaction below 1.

LTN evaluates a formula on every combination of the individuals that its variables range over, so a formula with v variables over n individuals builds a tensor with n^v entries. Problems with many constants and variables can run out of memory: about a third of the problems in the TPTP library would need more than a million entries.

Every call to `grounding.train` builds a new TensorFlow graph, which takes a while on a large problem. If you call it repeatedly, for example to log between calls, `grounding.train(10, reuse_graph=True)` keeps the graph and the optimizer. That also changes the results, because the optimizer's state carries on. An explicit `statements` list matching the default axioms reuses the same graph and optimizer. Use the flag only if you are sure that the graph hasn't changed: a kept graph sees new values of constants and weights, but it silently ignores a new predicate, connective or domain, and the other things listed in the docstring of `Grounding.train`.

### Files

`tptp.parse_file` reads a file and the files it includes. An include is looked up next to the file that has it, then in the folder given as `root`, which defaults to the environment variable `TPTP`. Problems in the TPTP library include axiom files by names such as `'Axioms/GRP003-0.ax'`, so pass the folder of the library:

```python
problem = tptp.parse_file("TPTP-v9.3.1/Problems/GRP/GRP001-1.p", root="TPTP-v9.3.1")
```

The statements of an included file take the place of its include, every time. So a file that is included twice, for example by the problem and again by one of its axiom files, gives its statements twice, and `sat()` counts them twice.

### Errors

`tptp.parse` and `tptp.parse_file` raise `SyntaxError` if the text is not valid TPTP, and `NotImplementedError` if it is valid TPTP that the reader doesn't support, such as the `tff` and `thf` languages, or a formula nested too deeply for Python's recursion limit. `tptp.parse` also refuses `include`, which needs `parse_file`, and `parse_file` raises `FileNotFoundError` if it can't find an included file. Queries that need a symbol with no grounding raise `ValueError`; comparisons between two interpreted literals need no grounding.

Annotations are discarded after their syntax is parsed. Formula data in annotations supports `$fof`, `$cnf` and `$fot`; `$tff` and `$thf` formula data still raises `NotImplementedError`.

## Adding a syntax

Each syntax only has to turn text into the syntax tree:

- `tree.py` -- the syntax tree, which all syntaxes share,
- `grounding.py` -- turns a syntax tree into LTN objects, whatever syntax it was read from,
- `tptp.py` and `tptp.lark` -- the TPTP parser and its grammar.

To add a syntax, add a module with a `parse(text)` function that returns a `Problem`, and a function that reads a single formula, which `Grounding` uses for queries given as text. If the syntax has constants that it interprets as themselves, like TPTP's numbers, set their `Fn.value`. `Problem` collects these in `Problem.values`, and queries can resolve other spellings of the same value. Also supply the optional `Problem.interpret_constant` callback to recognize interpreted aliases added directly to `grounding.constants`: given a valid constant name, it returns its interpreted value, or `None` for an uninterpreted name. This lets alias validation cover names outside the parsed problem without extending its domain. Don't import the syntax from `ltn/parser/__init__.py`, so that only people who use it need its dependencies.
