

from __future__ import annotations
from dataclasses import dataclass
from typing import FrozenSet, Set


# ── Base ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Formula:
    """Abstract base for all formula types."""

    def atoms(self) -> FrozenSet[str]:
        raise NotImplementedError

    def complexity(self) -> int:
        """Count of connectives — used as a proxy for proof difficulty."""
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


# ── Leaves ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Atom(Formula):
    """A propositional variable, e.g. P, Q, R."""
    name: str

    def atoms(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def complexity(self) -> int:
        return 0

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Top(Formula):
    """Logical truth (⊤)."""

    def atoms(self) -> FrozenSet[str]:
        return frozenset()

    def complexity(self) -> int:
        return 0

    def __str__(self) -> str:
        return "⊤"


@dataclass(frozen=True)
class Bot(Formula):
    """Logical falsity (⊥)."""

    def atoms(self) -> FrozenSet[str]:
        return frozenset()

    def complexity(self) -> int:
        return 0

    def __str__(self) -> str:
        return "⊥"


# ── Connectives ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Not(Formula):
    """Negation: ¬φ"""
    operand: Formula

    def atoms(self) -> FrozenSet[str]:
        return self.operand.atoms()

    def complexity(self) -> int:
        return 1 + self.operand.complexity()

    def __str__(self) -> str:
        inner = str(self.operand)
        if isinstance(self.operand, Atom):
            return f"¬{inner}"
        return f"¬({inner})"


@dataclass(frozen=True)
class And(Formula):
    """Conjunction: φ ∧ ψ"""
    left: Formula
    right: Formula

    def atoms(self) -> FrozenSet[str]:
        return self.left.atoms() | self.right.atoms()

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or(Formula):
    """Disjunction: φ ∨ ψ"""
    left: Formula
    right: Formula

    def atoms(self) -> FrozenSet[str]:
        return self.left.atoms() | self.right.atoms()

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies(Formula):
    """Implication: φ → ψ"""
    antecedent: Formula
    consequent: Formula

    def atoms(self) -> FrozenSet[str]:
        return self.antecedent.atoms() | self.consequent.atoms()

    def complexity(self) -> int:
        return 1 + self.antecedent.complexity() + self.consequent.complexity()

    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"


@dataclass(frozen=True)
class Iff(Formula):
    """Biconditional: φ ↔ ψ"""
    left: Formula
    right: Formula

    def atoms(self) -> FrozenSet[str]:
        return self.left.atoms() | self.right.atoms()

    def complexity(self) -> int:
        return 2 + self.left.complexity() + self.right.complexity()

    def __str__(self) -> str:
        return f"({self.left} ↔ {self.right})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def substitute(formula: Formula, var: str, replacement: Formula) -> Formula:
    """Replace every occurrence of atom `var` with `replacement`."""
    if isinstance(formula, Atom):
        return replacement if formula.name == var else formula
    if isinstance(formula, Not):
        return Not(substitute(formula.operand, var, replacement))
    if isinstance(formula, And):
        return And(substitute(formula.left, var, replacement),
                   substitute(formula.right, var, replacement))
    if isinstance(formula, Or):
        return Or(substitute(formula.left, var, replacement),
                  substitute(formula.right, var, replacement))
    if isinstance(formula, Implies):
        return Implies(substitute(formula.antecedent, var, replacement),
                       substitute(formula.consequent, var, replacement))
    if isinstance(formula, Iff):
        return Iff(substitute(formula.left, var, replacement),
                   substitute(formula.right, var, replacement))
    return formula


def is_tautology(formula: Formula) -> bool:
    """Check if formula is a tautology by truth-table evaluation."""
    atom_list = sorted(formula.atoms())
    n = len(atom_list)
    for mask in range(1 << n):
        assignment = {atom_list[i]: bool((mask >> i) & 1) for i in range(n)}
        if not evaluate(formula, assignment):
            return False
    return True


def evaluate(formula: Formula, assignment: dict) -> bool:
    """Evaluate a formula under a truth assignment."""
    if isinstance(formula, Atom):
        return assignment.get(formula.name, False)
    if isinstance(formula, Top):
        return True
    if isinstance(formula, Bot):
        return False
    if isinstance(formula, Not):
        return not evaluate(formula.operand, assignment)
    if isinstance(formula, And):
        return evaluate(formula.left, assignment) and evaluate(formula.right, assignment)
    if isinstance(formula, Or):
        return evaluate(formula.left, assignment) or evaluate(formula.right, assignment)
    if isinstance(formula, Implies):
        return (not evaluate(formula.antecedent, assignment)) or evaluate(formula.consequent, assignment)
    if isinstance(formula, Iff):
        return evaluate(formula.left, assignment) == evaluate(formula.right, assignment)
    raise ValueError(f"Unknown formula type: {type(formula)}")
