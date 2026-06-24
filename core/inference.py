

from __future__ import annotations
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Callable

from core.formula import (
    Formula, Atom, Top, Bot, Not, And, Or, Implies, Iff
)


# ── Rule data ─────────────────────────────────────────────────────────────────

@dataclass
class Rule:
    name: str
    description: str
    apply: Callable[[FrozenSet[Formula], Formula], List[FrozenSet[Formula]]]


# ── Individual rules ──────────────────────────────────────────────────────────

def _modus_ponens(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A, A→B  ⊢  B"""
    results = []
    for f in premises:
        if isinstance(f, Implies):
            if f.antecedent in premises:
                new_premises = premises | {f.consequent}
                if new_premises != premises:
                    results.append(new_premises)
    return results


def _modus_tollens(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """¬B, A→B  ⊢  ¬A"""
    results = []
    for f in premises:
        if isinstance(f, Implies):
            neg_consequent = Not(f.consequent)
            # check both ¬B and ¬¬B∈premises forms
            if neg_consequent in premises:
                new_fact = Not(f.antecedent)
                new_premises = premises | {new_fact}
                if new_premises != premises:
                    results.append(new_premises)
    return results


def _and_intro(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A, B  ⊢  A∧B  — only when the goal is a conjunction"""
    results = []
    if isinstance(goal, And):
        if goal.left in premises and goal.right in premises:
            results.append(premises | {goal})
    # Also introduce any conjunction that might help
    plist = list(premises)
    for i, a in enumerate(plist):
        for b in plist[i+1:]:
            conj = And(a, b)
            if conj not in premises:
                results.append(premises | {conj})
    return results[:4]  # cap to avoid explosion


def _and_elim(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A∧B  ⊢  A and B"""
    results = []
    for f in premises:
        if isinstance(f, And):
            new = premises | {f.left, f.right}
            if new != premises:
                results.append(new)
    return results


def _or_intro(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A  ⊢  A∨B — only when goal is a disjunction"""
    results = []
    if isinstance(goal, Or):
        if goal.left in premises:
            results.append(premises | {goal})
        if goal.right in premises:
            results.append(premises | {goal})
    return results


def _or_elim(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A∨B, A→C, B→C  ⊢  C"""
    results = []
    for f in premises:
        if isinstance(f, Or):
            a, b = f.left, f.right
            a_to_c = Implies(a, goal)
            b_to_c = Implies(b, goal)
            if a_to_c in premises and b_to_c in premises:
                results.append(premises | {goal})
    return results


def _hyp_syll(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A→B, B→C  ⊢  A→C"""
    results = []
    for f in premises:
        if isinstance(f, Implies):
            for g in premises:
                if isinstance(g, Implies) and f.consequent == g.antecedent:
                    new_fact = Implies(f.antecedent, g.consequent)
                    if new_fact not in premises:
                        results.append(premises | {new_fact})
    return results[:4]


def _double_neg(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """¬¬A  ⊢  A"""
    results = []
    for f in premises:
        if isinstance(f, Not) and isinstance(f.operand, Not):
            new_fact = f.operand.operand
            if new_fact not in premises:
                results.append(premises | {new_fact})
    return results


def _contradiction(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A, ¬A  ⊢  anything (ex falso)"""
    for f in premises:
        neg_f = Not(f)
        if neg_f in premises:
            # From contradiction we can derive the goal directly
            return [premises | {goal, Bot()}]
    return []


def _impl_intro(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """If goal is A→B and B follows from premises+{A}, introduce A→B."""
    if not isinstance(goal, Implies):
        return []
    antecedent = goal.antecedent
    consequent = goal.consequent
    # Assume antecedent and check if consequent is derivable in one step
    augmented = premises | {antecedent}
    # Try modus ponens in augmented set
    for f in augmented:
        if isinstance(f, Implies) and f.antecedent in augmented:
            if f.consequent == consequent:
                return [premises | {goal}]
    # If consequent already in premises
    if consequent in premises:
        return [premises | {goal}]
    return []


def _iff_elim(premises: FrozenSet[Formula], goal: Formula) -> List[FrozenSet[Formula]]:
    """A↔B  ⊢  A→B and B→A"""
    results = []
    for f in premises:
        if isinstance(f, Iff):
            lr = Implies(f.left, f.right)
            rl = Implies(f.right, f.left)
            new = premises | {lr, rl}
            if new != premises:
                results.append(new)
    return results


# ── Rule registry ─────────────────────────────────────────────────────────────

ALL_RULES: List[Rule] = [
    Rule("modus_ponens",       "A, A→B ⊢ B",             _modus_ponens),
    Rule("modus_tollens",      "¬B, A→B ⊢ ¬A",           _modus_tollens),
    Rule("and_intro",          "A, B ⊢ A∧B",             _and_intro),
    Rule("and_elim",           "A∧B ⊢ A, B",             _and_elim),
    Rule("or_intro",           "A ⊢ A∨B",                _or_intro),
    Rule("or_elim",            "A∨B, A→C, B→C ⊢ C",     _or_elim),
    Rule("hyp_syll",           "A→B, B→C ⊢ A→C",        _hyp_syll),
    Rule("double_neg",         "¬¬A ⊢ A",                _double_neg),
    Rule("contradiction",      "A, ¬A ⊢ anything",       _contradiction),
    Rule("impl_intro",         "assume A, derive B ⊢ A→B", _impl_intro),
    Rule("iff_elim",           "A↔B ⊢ A→B, B→A",        _iff_elim),
]

RULE_NAMES = [r.name for r in ALL_RULES]
NUM_RULES = len(ALL_RULES)


def apply_rule(rule_idx: int,
               premises: FrozenSet[Formula],
               goal: Formula) -> List[FrozenSet[Formula]]:
    """Apply rule by index. Returns list of possible new premise sets."""
    return ALL_RULES[rule_idx].apply(premises, goal)


def goal_reached(premises: FrozenSet[Formula], goal: Formula) -> bool:
    """Check if goal is in premises (proof complete)."""
    return goal in premises


def is_contradiction(premises: FrozenSet[Formula]) -> bool:
    """Check if premises are contradictory (Bot derived)."""
    return Bot() in premises
