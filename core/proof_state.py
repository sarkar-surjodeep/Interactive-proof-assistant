

from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Dict, List, Optional
import json

from core.formula import Formula, Implies
from core.parser import parse


# ── Proof state ───────────────────────────────────────────────────────────────

@dataclass
class ProofState:
    
    premises: FrozenSet[Formula]
    goal: Formula
    steps: List[str] = field(default_factory=list)
    depth: int = 0

    def with_new_premises(self, new_premises: FrozenSet[Formula], step_desc: str) -> "ProofState":
        return ProofState(
            premises=new_premises,
            goal=self.goal,
            steps=self.steps + [step_desc],
            depth=self.depth + 1,
        )

    def is_proved(self) -> bool:
        from core.inference import goal_reached
        return goal_reached(self.premises, self.goal)

    def premise_complexity(self) -> int:
        """Total complexity of all premises — decreasing means progress."""
        return sum(p.complexity() for p in self.premises)

    def goal_distance(self) -> int:
        
        if self.is_proved():
            return 0
        # Count how many atoms in goal are not yet known
        goal_atoms = self.goal.atoms()
        known_atoms = set()
        for p in self.premises:
            known_atoms |= p.atoms()
        return len(goal_atoms - known_atoms) + self.goal.complexity()

    def __repr__(self) -> str:
        premise_strs = ", ".join(str(p) for p in sorted(str(x) for x in self.premises))
        return f"ProofState(premises=[{premise_strs}], goal={self.goal})"


# ── Lemma library ─────────────────────────────────────────────────────────────

@dataclass
class LemmaEntry:
    formula: Formula
    proof_steps: List[str]
    times_used: int = 0
    episode_proved: int = 0

    def to_dict(self) -> dict:
        return {
            "formula": str(self.formula),
            "proof_steps": self.proof_steps,
            "times_used": self.times_used,
            "episode_proved": self.episode_proved,
        }


class LemmaLibrary:
    

    def __init__(self):
        self._lemmas: Dict[str, LemmaEntry] = {}
        self.episode_count = 0

    def add(self, formula: Formula, proof_steps: List[str]) -> bool:
        """Add a proved lemma. Returns True if it was new."""
        key = str(formula)
        if key not in self._lemmas:
            self._lemmas[key] = LemmaEntry(
                formula=formula,
                proof_steps=proof_steps,
                episode_proved=self.episode_count,
            )
            return True
        return False

    def contains(self, formula: Formula) -> bool:
        return str(formula) in self._lemmas

    def get(self, formula: Formula) -> Optional[LemmaEntry]:
        return self._lemmas.get(str(formula))

    def use(self, formula: Formula):
        """Record that a lemma was used in a proof."""
        entry = self.get(formula)
        if entry:
            entry.times_used += 1

    def all_lemmas(self) -> List[LemmaEntry]:
        return list(self._lemmas.values())

    def most_used(self, n: int = 5) -> List[LemmaEntry]:
        return sorted(self._lemmas.values(), key=lambda e: -e.times_used)[:n]

    def recent(self, n: int = 5) -> List[LemmaEntry]:
        return sorted(self._lemmas.values(), key=lambda e: -e.episode_proved)[:n]

    def size(self) -> int:
        return len(self._lemmas)

    def formulae(self) -> List[Formula]:
        return [e.formula for e in self._lemmas.values()]

    def to_context_string(self) -> str:
        """Human-readable summary for the LLM manager prompt."""
        if not self._lemmas:
            return "Library is empty."
        lines = []
        for entry in self.recent(10):
            lines.append(f"  • {entry.formula}  (used {entry.times_used}×)")
        return "\n".join(lines)

    def save(self, path: str):
        data = {k: v.to_dict() for k, v in self._lemmas.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        try:
            with open(path) as f:
                data = json.load(f)
            for key, v in data.items():
                formula = parse(
                    v["formula"]
                    .replace("¬", "~")
                    .replace("∧", "&")
                    .replace("∨", "|")
                    .replace("→", "->")
                    .replace("↔", "<->")
                    .replace("⊤", "True")
                    .replace("⊥", "False")
                )
                self._lemmas[key] = LemmaEntry(
                    formula=formula,
                    proof_steps=v["proof_steps"],
                    times_used=v["times_used"],
                    episode_proved=v["episode_proved"],
                )
        except FileNotFoundError:
            pass
