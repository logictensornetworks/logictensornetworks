# AGENTS.md

Guidance for AI coding agents (and humans) working on LTN (Logic Tensor Networks).

## Setup and commands

The project uses [uv](https://docs.astral.sh/uv/). Dependencies are declared in
`pyproject.toml` (core, the `examples` extra, and the `dev` group) and locked in `uv.lock`.

```bash
uv sync                                    # install dependencies (add --extra examples for examples/,
                                           # --extra parser for the parser tests)
uv run pytest test                         # run the test suite
uv run pytest test/test_core.py -k name    # run a subset
uv run ruff check --fix .                  # lint (the rule set is in pyproject.toml)
uv run ruff format .                       # format
```

CI fails if `ruff check .` or `ruff format --check .` report anything. `examples/` and
`tutorials/` are not linted.

After changing dependencies in `pyproject.toml`, run `uv lock` and commit `uv.lock`.

CI (`.github/workflows/tests.yml`) runs the tests on Python 3.10 to 3.13 with the latest
TensorFlow, and with TensorFlow 2.15 (the last release with Keras 2), on Linux and macOS.

## How to use LTN

Read the tutorials in `tutorials/` (in order, see `tutorials/README.md`) before writing examples,
tests or benchmarks that call LTN. The parser's syntax and conventions are in
`ltn/parser/README.md`.

- **Variables and constants** (`tutorials/1-grounding_non_logical_symbols.ipynb`): a
  `ltn.Variable` is a domain of individuals (its first axis), and LTN broadcasts over it: the
  output of a predicate or function has one axis per free variable. A `ltn.Constant` is a single
  individual and adds no axis. A data batch given as a `Variable` is a set of individuals that
  quantifiers aggregate over.
- **Diagonal quantification** (`tutorials/2-grounding_connectives.ipynb`): by default, several
  variables in one expression are combined as a cartesian product. Use `ltn.diag(x, y)` to pair
  their individuals one-to-one (e.g. images with their labels), which requires the same number of
  individuals in each.
- **Operators and gradients** (`tutorials/2b-operators_and_gradients.ipynb`): the choice of
  fuzzy operators (and their `p` exponents) changes what is learnable. Use the stable product
  configuration, as in the examples, rather than picking operators ad hoc.
- **Predicates from logits**: prefer `ltn.Predicate.FromLogits(model, activation_function=...)`
  over wrapping a sigmoid/softmax model yourself; `ltn.log` relies on it to compute in log space.

## Coding principles

Readability comes first. When in doubt, pick the version that is easiest to read.

- **KISS**: prefer the simplest solution that works. Avoid clever code and needless indirection.
- **YAGNI**: build what is needed now. No speculative abstractions, options, or extension
  points for hypothetical future use.
- **Locality of behavior**: keep the code that defines a behavior close to where it is used,
  so a reader can understand a unit by looking at it, without jumping across many files.

These principles also apply to requests, not just code. When a feature request or task could be
solved more simply another way, or seems unlikely to be needed, say so before implementing it,
so the maintainer can confirm or adjust the plan. Raise it briefly, don't refuse: the final call
is the maintainer's.

### Python version

The minimum supported version is Python 3.10 (`requires-python` in `pyproject.toml`), so
modern type hints (`int | None`, `list[str]`) can be used directly. Code must work with both
Keras 2 (TensorFlow 2.15) and Keras 3.

### Docstrings and types

- Use type hints on function signatures.
- Every public function or class gets a docstring, but keep it minimal: if the types and
  names already make the meaning clear, a one-line docstring is enough.
- Add longer explanations (and `Args:` / `Returns:` sections, Google style) only when they
  tell the reader something the signature does not, e.g. the reasoning behind an operator or
  a link to the relevant part of the LTN paper.

### Tests

Tests live in `test/`, written as `unittest.TestCase` classes and run with pytest. Not every change needs its own dedicated test. Prefer
higher-level tests where they fit: check what a formula evaluates to (its truth values and the
shape given by its free variables) rather than unit-testing every internal function.

## Branches

Work happens on a branch off `master`. Small documentation-only changes (e.g. to AGENTS.md) may
be committed directly to `master`. Name branches by kind:

- `feat/<name>`: new feature
- `bug/<name>`: bug fix
- `test/<name>`: tests only
- `ref/<name>`: refactoring
- `docs/<name>`: documentation only
- `misc/<name>`: anything else

## Issues and pull requests

- **One PR addresses one issue.** Several fixes may be bundled only when they are minor
  (typos, small cleanups in the same area); mention them in the description. Split unrelated
  non-trivial changes into separate PRs.
- **Log bugs you run into in an issue.** If you find a bug while working on something else,
  don't hesitate to open an issue for it, with a minimal reproduction, so it is recorded for
  later. It doesn't need to be fixed right away, nor in the current PR.
- Link the issue in the PR body with `Closes #N`.
- **Tag issues and PRs with the repository's labels** when you create or address them. For
  example, use `bug` for a bug fix and `enhancement` for a new feature. Pick from the existing
  labels (`gh label list`) rather than creating new ones.
- PRs are squash-merged, so the PR title becomes the commit message on `master`: make it a
  clear summary of the change.
- PR descriptions cover:
  - **Why**: the problem, or why the change was needed.
  - **How**: how it was solved, and alternatives considered, if any.
  - **Testing & risks**: how to verify it, and what could break.
  A one-line answer per section is fine for small changes.
- **Show user-visible changes with a short example.** For API changes, a snippet of the code
  before and after, how it's called, and what it returns. For bug fixes and behavior changes, a
  minimal reproduction of the old and new behavior. Internal refactors, CI and formatting
  changes don't need one. Linking to a test or example script in the PR is fine too. Examples
  must actually run: paste the real output, don't write it from memory.

## Docs and reports

There are two kinds of documentation. Choose by who will read it.

- **User docs** (`README.md`, `tutorials/`, `examples/`, and READMEs inside subpackages such as
  `ltn/parser/README.md`) are **user-facing**. They explain features and behaviors that anyone
  using the library should know: how to ground symbols, what each operator computes, how to
  train and query a knowledge base. Write them for someone who calls LTN, without assuming they
  have read its source.
- **Reports** (`reports/`, Markdown) are **internal-facing**. They explain how a subpackage or
  feature works inside: its design and data flow, its limitations and known bugs, with
  references to files and lines. They are for maintainers, human or AI. We will ask when a
  report is needed.

When a topic has both sides, put the user-facing behavior in the user docs and link to it from
the report, rather than explaining it twice. In both, examples must run and show real output.

Before answering questions about the code, list `reports/` (if it exists) for files that may be
useful.
