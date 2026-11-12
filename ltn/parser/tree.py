"""The syntax tree, which every syntax reads text into, and `Grounding` turns into LTN objects."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

AXIOM_ROLES = (
    "axiom",
    "hypothesis",
    "definition",
    "assumption",
    "lemma",
    "theorem",
    "corollary",
    "plain",
)
CONJECTURE_ROLES = ("conjecture", "negated_conjecture")


@dataclasses.dataclass(frozen=True)
class Var:
    name: str

@dataclasses.dataclass(frozen=True)
class Fn:
    """Function term. A constant is a function without arguments."""
    name: str
    args: tuple = ()

@dataclasses.dataclass(frozen=True)
class Atom:
    """Predicate atom. Equality is the predicate "="."""
    pred: str
    args: tuple = ()

@dataclasses.dataclass(frozen=True)
class Truth:
    value: bool

@dataclasses.dataclass(frozen=True)
class Not:
    arg: object

@dataclasses.dataclass(frozen=True)
class BinOp:
    op: str  # "=>" or "<=>"
    left: object
    right: object

@dataclasses.dataclass(frozen=True)
class Chain:
    """Two or more operands of & or |, which group to the left."""
    op: str  # "&" or "|"
    args: tuple

@dataclasses.dataclass(frozen=True)
class Quant:
    kind: str  # "!" (forall) or "?" (exists)
    vars: tuple
    body: object

@dataclasses.dataclass(frozen=True)
class Refused:
    """A part of a formula that the reader refuses. Raising only when the formula of a statement
    is checked means that the same thing in an annotation, which has no effect, is not refused."""
    error: Exception

@dataclasses.dataclass(frozen=True)
class Statement:
    name: str
    role: str
    formula: object


def free_vars(f, bound=frozenset()) -> list[str]:
    if isinstance(f, Refused):
        raise f.error
    if isinstance(f, Var):
        return [] if f.name in bound else [f.name]
    if isinstance(f, (Fn, Atom, Chain)):
        return list(dict.fromkeys(v for a in f.args for v in free_vars(a, bound)))
    if isinstance(f, Not):
        return free_vars(f.arg, bound)
    if isinstance(f, BinOp):
        return list(dict.fromkeys(free_vars(f.left, bound) + free_vars(f.right, bound)))
    if isinstance(f, Quant):
        return free_vars(f.body, bound | set(f.vars))
    return []


@dataclasses.dataclass
class Problem:
    """The statements of a problem, its symbols with their arities, and how to read a query.

    `values` holds the constants that a syntax interprets as themselves, such as TPTP's numbers and
    distinct objects. Two of them are the same individual if their values are equal, and different
    individuals otherwise. The syntax fills it in."""
    statements: list[Statement]
    parse_formula: Callable[[str], object]
    predicates: dict[str, int] = dataclasses.field(default_factory=dict)
    functions: dict[str, int] = dataclasses.field(default_factory=dict)
    values: dict[str, object] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for statement in self.statements:
            self._collect(statement.formula)

    def _collect(self, f) -> None:
        if isinstance(f, (Atom, Fn)):
            table = self.predicates if isinstance(f, Atom) else self.functions
            name = f.pred if isinstance(f, Atom) else f.name
            if name != "=" and table.setdefault(name, len(f.args)) != len(f.args):
                raise NotImplementedError(f"Symbol {name} is used with different arities.")
            for arg in f.args:
                self._collect(arg)
        elif isinstance(f, Not):
            self._collect(f.arg)
        elif isinstance(f, Chain):
            for arg in f.args:
                self._collect(arg)
        elif isinstance(f, BinOp):
            self._collect(f.left)
            self._collect(f.right)
        elif isinstance(f, Quant):
            self._collect(f.body)

    def with_role(self, *roles: str) -> list[Statement]:
        return [s for s in self.statements if s.role in roles]
