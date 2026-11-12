"""Reads TPTP problems in the FOF and CNF languages. The grammar is in tptp.lark.

`parse` reads text, and `parse_file` reads a file together with the files it includes.
"""
from __future__ import annotations

import dataclasses
import fractions
import os
import pathlib
import re
import sys

try:
    import lark
except ImportError as error:
    raise ImportError(
        "ltn.parser.tptp needs lark. Install it with: pip install ltn[parser]"
    ) from error

from ltn.parser.tree import (
    AXIOM_ROLES,
    CONJECTURE_ROLES,
    Atom,
    BinOp,
    Chain,
    Fn,
    Not,
    Problem,
    Quant,
    Refused,
    Statement,
    Truth,
    Var,
    free_vars,
)

# The grammar lists `unknown` as a role, but calls it an error situation.
UNSUPPORTED_ROLES = ("type", "interpretation", "logic", "unknown")

_NONASSOC_CONNECTIVES = {
    "=>": lambda left, right: BinOp("=>", left, right),
    "<=": lambda left, right: BinOp("=>", right, left),
    "<=>": lambda left, right: BinOp("<=>", left, right),
    "<~>": lambda left, right: Not(BinOp("<=>", left, right)),
    "~|": lambda left, right: Not(Chain("|", (left, right))),
    "~&": lambda left, right: Not(Chain("&", (left, right))),
}


@dataclasses.dataclass(frozen=True)
class Include:
    """An include directive. `names` is None to include every statement of the file."""
    file: str
    names: tuple | None
    line: int


@lark.v_args(inline=True)
class _SyntaxTree(lark.Transformer):
    """Builds the syntax tree while tptp.lark is parsed, one method per rule. This is also where
    the rules that a context-free grammar cannot express are applied."""

    def tptp_file(self, *items) -> list:
        return list(items)

    def fof_annotated(self, name, role, formula, *annotations) -> Statement:
        free = free_vars(formula)
        if free:
            raise SyntaxError(
                f"All variables of a fof formula must be quantified: {free} in {name}."
            )
        return Statement(name, role, formula)

    def cnf_annotated(self, name, role, formula, *annotations) -> Statement:
        free = free_vars(formula)
        if free:  # the variables of a clause are implicitly universally quantified
            formula = Quant("!", tuple(free), formula)
        return Statement(name, role, formula)

    def include(self, keyword, file_name, names=None, space=None) -> Include:
        if space is not None:
            raise NotImplementedError(
                f"include(...) with a space name at line {keyword.line} is not supported."
            )
        if file_name.startswith("'"):
            file_name = re.sub(r"\\(['\\])", r"\1", file_name[1:-1])
        return Include(file_name, names, keyword.line)

    def formula_selection(self, *names) -> tuple | None:
        return names or None  # * selects every statement

    def unsupported_language(self, language):
        raise NotImplementedError(
            f"{language}(...) at line {language.line} is not supported, only fof(...) and cnf(...)."
        )

    def unsupported_formula_data(self, language):
        raise NotImplementedError(
            f"{language}(...) at line {language.line} is not supported, "
            "only $fof(...) and $cnf(...)."
        )

    def formula_role(self, word, *subrole) -> str:
        if word not in AXIOM_ROLES + CONJECTURE_ROLES + UNSUPPORTED_ROLES:
            raise SyntaxError(f"Expected a formula role at line {word.line}, got {str(word)!r}.")
        if subrole or word in UNSUPPORTED_ROLES:
            raise NotImplementedError(f"Role {word} at line {word.line} is not supported.")
        return str(word)

    def name(self, word) -> str:
        return str(word)

    def plain_word(self, word) -> str:
        if word.type == "SINGLE_QUOTED" and re.fullmatch(r"'[a-z][A-Za-z0-9_]*'", word):
            return word[1:-1]  # 'cat' and cat are the same word, unlike '123' and 123
        return str(word)

    def fof_binary_pair(self, left, connective, right):
        return _NONASSOC_CONNECTIVES[connective](left, right)

    def fof_or_formula(self, *operands) -> Chain:
        return Chain("|", operands)

    def fof_and_formula(self, *operands) -> Chain:
        return Chain("&", operands)

    def negation(self, formula) -> Not:
        return Not(formula)

    def fof_quantified_formula(self, quantifier, variables, body) -> Quant:
        return Quant(str(quantifier), variables, body)

    def fof_variable_list(self, *variables) -> tuple:
        return tuple(str(variable) for variable in variables)

    def fof_plain_atomic_formula(self, term: Fn) -> Atom:
        return Atom(term.name, term.args)

    def fof_defined_infix_formula(self, left, right) -> Atom:
        return Atom("=", (left, right))

    def fof_infix_unary(self, left, right) -> Not:
        return Not(Atom("=", (left, right)))

    def true_formula(self) -> Truth:
        return Truth(True)

    def false_formula(self) -> Truth:
        return Truth(False)

    def fof_plain_term(self, functor, *args) -> Fn:
        return Fn(functor, args)

    def defined_term(self, word) -> Fn:
        return Fn(str(word))

    def variable(self, word) -> Var:
        return Var(str(word))

    def fof_unsupported_term(self, word, *args) -> Refused:
        if word in ("$true", "$false"):
            return Refused(SyntaxError(f"{word} at line {word.line} is a formula, not a term."))
        error = f"Symbols such as {word} at line {word.line} are not supported."
        return Refused(NotImplementedError(error))

    def fof_sequent(self, *parts) -> Refused:
        return Refused(NotImplementedError("Sequents are not supported."))


_parser = lark.Lark.open(
    "tptp.lark",
    rel_to=__file__,
    parser="lalr",
    start=["tptp_file", "fof_logic_formula"],
    transformer=_SyntaxTree(),
)


def _token_name(terminal: str) -> str:
    if terminal == "$END":
        return "the end of the file"
    pattern = _parser.get_terminal(terminal).pattern
    if isinstance(pattern, lark.lexer.PatternStr):
        return repr(pattern.value)
    return terminal.lower().replace("_", " ")


def _read(text: str, start: str):
    try:
        return _parser.parse(text, start=start)
    except lark.UnexpectedInput as error:
        expected = error.accepts if isinstance(error, lark.UnexpectedToken) else error.allowed
        names = ", ".join(sorted(map(_token_name, expected)))
        raise SyntaxError(
            f"Invalid TPTP at line {error.line}, column {error.column}, expected {names}:\n"
            f"{error.get_context(text)}"
        ) from None


def _guarded(action):
    """Runs `action`, refusing formulas that are nested too deeply for Python's recursion limit."""
    try:
        return action()
    except RecursionError:
        raise NotImplementedError(f"A formula is nested too deeply for Python's recursion limit "
                                  f"({sys.getrecursionlimit()}).") from None


def _value(name: str):
    """TPTP interprets numbers and distinct objects as themselves. Numbers are equal if their values
    are, so 1/2 = 0.5, and a number is never equal to a distinct object."""
    if name.startswith('"'):
        return "distinct object", name
    if name[0] in "+-0123456789":
        return "number", fractions.Fraction(name)
    return None


def _problem(statements: list[Statement]) -> Problem:
    problem = Problem(statements, parse_formula)
    for name, arity in problem.functions.items():
        value = _value(name) if arity == 0 else None
        if value is not None:
            problem.values[name] = value
    return problem


def parse(text: str) -> Problem:
    """Reads TPTP text. Raises SyntaxError if it is not valid TPTP, and NotImplementedError if it
    is valid TPTP that the reader doesn't support. That includes `include`, which needs
    `parse_file`."""
    items = _guarded(lambda: _read(text, "tptp_file"))
    for item in items:
        if isinstance(item, Include):
            raise NotImplementedError(f"include(...) at line {item.line} needs the file it names: "
                                      f"read the problem with tptp.parse_file.")
    return _guarded(lambda: _problem(items))


def parse_file(path, root=None) -> Problem:
    """Reads a TPTP file, with the statements of each included file in place of its include. An
    include is looked up next to the file that has it, then in `root`, which defaults to the
    environment variable TPTP. Problems in the TPTP library include axiom files by names such as
    'Axioms/SET001+0.ax', so `root` is the folder of the library. Raises FileNotFoundError if an
    included file isn't found, and otherwise the errors of `parse`, with the name of the file."""
    if root is None:
        root = os.environ.get("TPTP")
    statements = _guarded(lambda: _statements(pathlib.Path(path).resolve(), root, {}, ()))
    return _guarded(lambda: _problem(statements))


def _statements(path: pathlib.Path, root, cache: dict, including: tuple) -> list[Statement]:
    if path in including:
        chain = " includes ".join(map(str, (*including, path)))
        raise SyntaxError(f"A file includes itself: {chain}.")
    if path not in cache:
        try:
            cache[path] = _read(path.read_text(encoding="utf-8"), "tptp_file")
        except (SyntaxError, NotImplementedError) as error:
            raise type(error)(f"In {path}: {error}") from None
    statements = []
    for item in cache[path]:
        if isinstance(item, Statement):
            statements.append(item)
            continue
        included = _statements(_find(item, path, root), root, cache, (*including, path))
        if item.names is not None:
            missing = sorted(set(item.names) - {statement.name for statement in included})
            if missing:
                raise SyntaxError(
                    f"In {path}: the include at line {item.line} selects {', '.join(missing)}, "
                    f"which {item.file} doesn't have."
                )
            included = [statement for statement in included if statement.name in item.names]
        statements += included
    return statements


def _find(include: Include, path: pathlib.Path, root) -> pathlib.Path:
    places = [path.parent] + ([pathlib.Path(root)] if root else [])
    for place in places:
        candidate = place / include.file
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"In {path}: the file of include('{include.file}') at line {include.line} is in none "
        f"of {', '.join(map(str, places))}. For a problem of the TPTP library, pass root= or "
        "set the environment variable TPTP to the folder of the library."
    )


def parse_formula(text: str):
    """Reads a closed formula in fof syntax, for example a query."""
    formula = _guarded(lambda: _read(text, "fof_logic_formula"))
    if _guarded(lambda: free_vars(formula)):
        raise SyntaxError(f"All variables must be quantified: {free_vars(formula)}.")
    return formula
