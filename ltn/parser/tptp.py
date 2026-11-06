"""Reads TPTP problems in the FOF and CNF languages. The grammar is in tptp.lark."""
from __future__ import annotations

import re

try:
    import lark
except ImportError as error:
    raise ImportError("ltn.parser.tptp needs lark. Install it with: pip install ltn[parser]") from error

from ltn.parser.tree import (AXIOM_ROLES, CONJECTURE_ROLES, Atom, BinOp, Chain, Fn, Not, Problem, Quant, Refused,
                             Statement, Truth, Var, free_vars)

UNSUPPORTED_ROLES = ("type", "interpretation", "logic")

_NONASSOC_CONNECTIVES = {
    "=>": lambda left, right: BinOp("=>", left, right),
    "<=": lambda left, right: BinOp("=>", right, left),
    "<=>": lambda left, right: BinOp("<=>", left, right),
    "<~>": lambda left, right: Not(BinOp("<=>", left, right)),
    "~|": lambda left, right: Not(Chain("|", (left, right))),
    "~&": lambda left, right: Not(Chain("&", (left, right))),
}


@lark.v_args(inline=True)
class _SyntaxTree(lark.Transformer):
    """Builds the syntax tree while tptp.lark is parsed, one method per rule. This is also where
    the rules that a context-free grammar cannot express are applied."""

    def tptp_file(self, *statements) -> list[Statement]:
        return list(statements)

    def fof_annotated(self, name, role, formula, *annotations) -> Statement:
        free = free_vars(formula)
        if free:
            raise SyntaxError(f"All variables of a fof formula must be quantified: {free} in {name}.")
        return Statement(name, role, formula)

    def cnf_annotated(self, name, role, formula, *annotations) -> Statement:
        free = free_vars(formula)
        if free:  # the variables of a clause are implicitly universally quantified
            formula = Quant("!", tuple(free), formula)
        return Statement(name, role, formula)

    def unsupported_language(self, language):
        raise NotImplementedError(f"{language}(...) at line {language.line} is not supported, only fof(...) and cnf(...).")

    def unsupported_formula_data(self, language):
        raise NotImplementedError(f"{language}(...) at line {language.line} is not supported, only $fof(...) and $cnf(...).")

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
        return Refused(NotImplementedError(f"Symbols such as {word} at line {word.line} are not supported."))

    def fof_sequent(self, *parts) -> Refused:
        return Refused(NotImplementedError("Sequents are not supported."))


_parser = lark.Lark.open("tptp.lark", rel_to=__file__, parser="lalr", start=["tptp_file", "fof_logic_formula"],
                         transformer=_SyntaxTree())


def _token_name(terminal: str) -> str:
    if terminal == "$END":
        return "the end of the file"
    pattern = _parser.get_terminal(terminal).pattern
    return repr(pattern.value) if isinstance(pattern, lark.lexer.PatternStr) else terminal.lower().replace("_", " ")


def _read(text: str, start: str):
    try:
        return _parser.parse(text, start=start)
    except lark.UnexpectedInput as error:
        expected = error.accepts if isinstance(error, lark.UnexpectedToken) else error.allowed
        raise SyntaxError(f"Invalid TPTP at line {error.line}, column {error.column}, expected "
                          f"{', '.join(sorted(map(_token_name, expected)))}:\n{error.get_context(text)}") from None


def parse(text: str) -> Problem:
    """Reads a TPTP file. Raises SyntaxError if it is not valid TPTP, and NotImplementedError if
    it is valid TPTP that the reader doesn't support."""
    return Problem(_read(text, "tptp_file"), parse_formula)


def parse_formula(text: str):
    """Reads a closed formula in fof syntax, for example a query."""
    formula = _read(text, "fof_logic_formula")
    if free_vars(formula):
        raise SyntaxError(f"All variables must be quantified: {free_vars(formula)}.")
    return formula
