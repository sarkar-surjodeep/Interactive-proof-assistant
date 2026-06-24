

from __future__ import annotations
import re
from typing import List, Tuple

from core.formula import (
    Formula, Atom, Top, Bot, Not, And, Or, Implies, Iff
)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

TOKEN_PATTERNS = [
    ("IFF",     r"<->|↔"),
    ("IMPL",    r"->|→"),
    ("OR",      r"\||\∨"),
    ("AND",     r"&|∧"),
    ("NOT",     r"~|¬"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("TOP",     r"⊤|True|TRUE|true"),
    ("BOT",     r"⊥|False|FALSE|false"),
    ("ATOM",    r"[A-Za-z][A-Za-z0-9_]*"),
    ("SKIP",    r"\s+"),
]

TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_PATTERNS)
)


def tokenize(text: str) -> List[Tuple[str, str]]:
    tokens = []
    for m in TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind == "SKIP":
            continue
        tokens.append((kind, m.group()))
    return tokens


# ── Recursive descent parser ───────────────────────────────────────────────────

class Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos][0]
        return None

    def consume(self, expected: str | None = None) -> Tuple[str, str]:
        tok = self.tokens[self.pos]
        if expected and tok[0] != expected:
            raise SyntaxError(f"Expected {expected}, got {tok[0]} ('{tok[1]}')")
        self.pos += 1
        return tok

    def parse(self) -> Formula:
        f = self.parse_iff()
        if self.pos < len(self.tokens):
            raise SyntaxError(f"Unexpected token: '{self.tokens[self.pos][1]}'")
        return f

    def parse_iff(self) -> Formula:
        left = self.parse_impl()
        while self.peek() == "IFF":
            self.consume("IFF")
            right = self.parse_impl()
            left = Iff(left, right)
        return left

    def parse_impl(self) -> Formula:
        left = self.parse_or()
        while self.peek() == "IMPL":
            self.consume("IMPL")
            right = self.parse_or()
            left = Implies(left, right)
        return left

    def parse_or(self) -> Formula:
        left = self.parse_and()
        while self.peek() == "OR":
            self.consume("OR")
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self) -> Formula:
        left = self.parse_not()
        while self.peek() == "AND":
            self.consume("AND")
            right = self.parse_not()
            left = And(left, right)
        return left

    def parse_not(self) -> Formula:
        if self.peek() == "NOT":
            self.consume("NOT")
            operand = self.parse_not()
            return Not(operand)
        return self.parse_atom()

    def parse_atom(self) -> Formula:
        tok_type = self.peek()
        if tok_type == "TOP":
            self.consume("TOP")
            return Top()
        if tok_type == "BOT":
            self.consume("BOT")
            return Bot()
        if tok_type == "ATOM":
            _, name = self.consume("ATOM")
            return Atom(name)
        if tok_type == "LPAREN":
            self.consume("LPAREN")
            f = self.parse_iff()
            self.consume("RPAREN")
            return f
        raise SyntaxError(
            f"Unexpected token: '{self.tokens[self.pos][1]}'" if self.pos < len(self.tokens)
            else "Unexpected end of input"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def parse(text: str) -> Formula:
    """Parse a formula string into a Formula object."""
    tokens = tokenize(text)
    if not tokens:
        raise SyntaxError("Empty formula")
    return Parser(tokens).parse()


def parse_proof_state(premises_str: List[str], goal_str: str):
    """Convenience: parse a list of premise strings and a goal string."""
    premises = frozenset(parse(p) for p in premises_str)
    goal = parse(goal_str)
    return premises, goal
