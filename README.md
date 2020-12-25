# Logic Tensor Networks (LTN)

# Installation

Clone the LTN repository and install it using `pip install -e <local project path>`.

Following are the dependencies we used for development (similar versions should run fine):
- python 3.8
- tensorflow >= 2.2 (for running the core system)
- numpy >= 1.18 (for examples)
- matplotlib >= 3.2 (for examples)

# Repository structure

- `logictensornetworks/core.py` -- core system for defining constants, variables, predicates, functions and formulas,
- `logictensornetworks/fuzzy_ops.py` -- a collection of fuzzy logic operators defined using Tensorflow primitives,
- `logictensornetworks/utils.py` -- a collection of useful functions,
- `tutorials/` -- tutorials to start with LTN,
- `examples/` -- various problems approached using LTN,
- `tests/` -- tests.

# Getting Started

## Tutorials

`tutorials/` contains a walk-through of LTN. In order, the tutorials cover the following topics:
1. Grounding in LTN part 1: Real Logic, constants, predicates, functions, variables,
2. Grounding in LTN part 2: connectives and quantifiers (+ complement: choosing appropriate operators for learning),
3. Learning in LTN: using satisfiability of LTN formulas as a training objective,
4. Reasoning in LTN: measuring if a formula is the logical consequence of a knowledgebase.

The tutorials are implemented using jupyter notebooks.

## Examples

`examples/` contains a series of experiments. Their objective is to show how the language of Real Logic can be used to specify a number of tasks that involve learning from data and reasoning about logical knowledge. Examples of such tasks are: classification (`binary_classification`, `multiclass_classification`, `mnist`), regression, clustering, link prediction (`smokes_friends_cancer`, `parent_ancestor`).

The examples are presented with both jupyter notebooks and Python scripts.

# Acknowledgements

LTN has been developed thanks to active contributions and discussions with the following people (in alphabetical order):
- Alessandro Daniele (FBK)
- Artur d’Avila Garces (City)
- Benedikt Wagner (City)
- Francesco Giannini (UniSiena)
- Giuseppe Marra (UniSiena)
- Ivan Donadello (FBK)
- Lucas Brukberger (UniOsnabruck)
- Luciano Serafini (FBK)
- Marco Gori (UniSiena)
- Michael Spranger (Sony AI)
- Michelangelo Diligenti (UniSiena)
- Samy Badreddine (Sony AI)