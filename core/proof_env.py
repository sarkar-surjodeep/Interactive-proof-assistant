

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import FrozenSet, List, Optional, Tuple, Dict, Any

from core.formula import Formula, Atom, Not, And, Or, Implies, Iff, Top, Bot
from core.inference import ALL_RULES, NUM_RULES, apply_rule, goal_reached
from core.proof_state import ProofState, LemmaLibrary
from core.parser import parse


# ── Encoding ──────────────────────────────────────────────────────────────────

# Fixed vocabulary of atoms we care about (can be extended)
ATOM_VOCAB = ["P", "Q", "R", "S", "T", "A", "B", "C", "D", "E"]
MAX_PREMISES = 20
OBS_DIM = len(ATOM_VOCAB) * 3 + NUM_RULES + 10  # presence + rule mask + misc


def formula_to_vec(formula: Formula, vocab: List[str]) -> np.ndarray:
    """
    Crude but fast formula fingerprint:
    - Which atoms appear?
    - What connective types appear?
    - Complexity bucket
    """
    v = np.zeros(len(vocab) + 6, dtype=np.float32)
    atoms = formula.atoms()
    for i, name in enumerate(vocab):
        if name in atoms:
            v[i] = 1.0
    # Connective presence flags
    text = str(formula)
    v[len(vocab)]     = float("→" in text)
    v[len(vocab) + 1] = float("∧" in text)
    v[len(vocab) + 2] = float("∨" in text)
    v[len(vocab) + 3] = float("¬" in text)
    v[len(vocab) + 4] = float("↔" in text)
    v[len(vocab) + 5] = min(formula.complexity() / 10.0, 1.0)
    return v


def state_to_obs(state: ProofState, library: LemmaLibrary) -> np.ndarray:
    """Encode proof state as a fixed-size observation vector."""
    FEAT = len(ATOM_VOCAB) + 6

    # Aggregate over premises (mean pooling, capped at MAX_PREMISES)
    prem_vecs = [formula_to_vec(p, ATOM_VOCAB) for p in list(state.premises)[:MAX_PREMISES]]
    if prem_vecs:
        prem_agg = np.mean(prem_vecs, axis=0)
    else:
        prem_agg = np.zeros(FEAT, dtype=np.float32)

    goal_vec = formula_to_vec(state.goal, ATOM_VOCAB)

    # Rule applicability mask (which rules could fire)
    rule_mask = np.zeros(NUM_RULES, dtype=np.float32)
    for i in range(NUM_RULES):
        if apply_rule(i, state.premises, state.goal):
            rule_mask[i] = 1.0

    # Misc features
    misc = np.array([
        min(len(state.premises) / MAX_PREMISES, 1.0),
        min(state.depth / 50.0, 1.0),
        min(state.goal_distance() / 10.0, 1.0),
        min(library.size() / 50.0, 1.0),
    ], dtype=np.float32)

    return np.concatenate([prem_agg, goal_vec, rule_mask, misc])


# ── Environment ───────────────────────────────────────────────────────────────

class ProofEnv(gym.Env):
    """
    Worker-level environment.
    At each step the worker picks an inference rule to apply.
    The episode ends when the goal is proved or step budget is exhausted.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        theorems: List[Dict],        # list of {"premises": [...], "goal": "..."}
        library: LemmaLibrary,
        max_steps: int = 40,
        subgoal: Optional[Formula] = None,  # if set, worker aims for subgoal, not main goal
        render_mode: str = "human",
    ):
        super().__init__()
        self.theorems = theorems
        self.library = library
        self.max_steps = max_steps
        self.render_mode = render_mode

        obs_dim = (len(ATOM_VOCAB) + 6) * 2 + NUM_RULES + 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_RULES)

        self.state: Optional[ProofState] = None
        self.active_goal: Optional[Formula] = None
        self.step_count = 0
        self.prev_distance = 0
        self.current_theorem: Optional[Dict] = None

    def reset(self, seed=None, options=None, theorem_idx: Optional[int] = None):
        super().reset(seed=seed)

        if theorem_idx is not None:
            thm = self.theorems[theorem_idx]
        else:
            thm = self.theorems[np.random.randint(len(self.theorems))]

        self.current_theorem = thm
        premises = frozenset(parse(p) for p in thm["premises"])
        goal = parse(thm["goal"])

        # Inject library lemmas as extra premises (this is the reuse mechanism)
        for lemma in self.library.formulae():
            premises = premises | {lemma}

        self.state = ProofState(premises=premises, goal=goal)
        self.active_goal = goal
        self.step_count = 0
        self.prev_distance = self.state.goal_distance()

        return state_to_obs(self.state, self.library), {}

    def set_subgoal(self, subgoal: Formula):
        """Manager sets a lemma subgoal for the worker to pursue."""
        self.active_goal = subgoal
        # Create a sub-state with the same premises but new goal
        self.state = ProofState(
            premises=self.state.premises,
            goal=subgoal,
            steps=self.state.steps.copy(),
            depth=self.state.depth,
        )
        self.prev_distance = self.state.goal_distance()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.state is not None, "Call reset() first"

        rule = ALL_RULES[action]
        new_premise_sets = apply_rule(action, self.state.premises, self.state.goal)

        reward = 0.0
        terminated = False
        truncated = False
        info = {"rule": rule.name, "fired": False, "proved": False}

        if new_premise_sets:
            # Pick the premise set that most reduces goal distance
            best = min(new_premise_sets,
                       key=lambda ps: ProofState(ps, self.state.goal).goal_distance())
            step_desc = f"{rule.name}: added {best - self.state.premises}"
            self.state = self.state.with_new_premises(best, step_desc)

            new_distance = self.state.goal_distance()
            progress = self.prev_distance - new_distance
            reward += 0.1 + 0.2 * max(0, progress)
            self.prev_distance = new_distance
            info["fired"] = True
        else:
            reward -= 0.05  # wasted step

        self.step_count += 1

        if self.state.is_proved():
            reward += 5.0
            terminated = True
            info["proved"] = True
            self.library.add(self.state.goal, self.state.steps)

        if self.step_count >= self.max_steps:
            reward -= 1.0
            truncated = True

        obs = state_to_obs(self.state, self.library)
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self.state:
            print(f"\n--- Step {self.step_count} ---")
            print(f"Goal: {self.state.goal}")
            print(f"Premises ({len(self.state.premises)}):")
            for p in sorted(str(x) for x in self.state.premises):
                print(f"  {p}")
            if self.state.steps:
                print(f"Last step: {self.state.steps[-1]}")


# ── Theorem curriculum ────────────────────────────────────────────────────────

CURRICULUM: List[Dict] = [
    # Level 1 — single rule
    {"premises": ["P", "P -> Q"],         "goal": "Q",
     "name": "modus ponens",              "level": 1},
    {"premises": ["P", "Q"],              "goal": "P & Q",
     "name": "and introduction",          "level": 1},
    {"premises": ["P & Q"],               "goal": "P",
     "name": "and elimination left",      "level": 1},
    {"premises": ["P & Q"],               "goal": "Q",
     "name": "and elimination right",     "level": 1},
    {"premises": ["P"],                   "goal": "P | Q",
     "name": "or introduction",           "level": 1},
    {"premises": ["~~P"],                 "goal": "P",
     "name": "double negation",           "level": 1},

    # Level 2 — two rules chained
    {"premises": ["P", "P -> Q", "Q -> R"],  "goal": "R",
     "name": "chain of implications",        "level": 2},
    {"premises": ["P & Q", "Q -> R"],         "goal": "R",
     "name": "and-elim then mp",             "level": 2},
    {"premises": ["P -> Q", "Q -> R"],        "goal": "P -> R",
     "name": "hypothetical syllogism",       "level": 2},
    {"premises": ["~Q", "P -> Q"],            "goal": "~P",
     "name": "modus tollens",                "level": 2},
    {"premises": ["P", "Q", "Q -> R"],        "goal": "P & R",
     "name": "mp then and-intro",            "level": 2},

    # Level 3 — three or more rules
    {"premises": ["P -> Q", "R -> S", "P & R"],   "goal": "Q & S",
     "name": "parallel implications",             "level": 3},
    {"premises": ["P", "P -> Q", "P -> R"],        "goal": "Q & R",
     "name": "two consequences",                  "level": 3},
    {"premises": ["P | Q", "P -> R", "Q -> R"],    "goal": "R",
     "name": "or elimination",                    "level": 3},
    {"premises": ["P <-> Q", "P"],                 "goal": "Q",
     "name": "biconditional use",                 "level": 3},

    # Level 4 — requiring lemma decomposition
    {"premises": ["P -> Q", "Q -> R", "R -> S", "P"],  "goal": "S",
     "name": "chain of four",                          "level": 4},
    {"premises": ["P & Q", "P -> R", "Q -> S"],         "goal": "R & S",
     "name": "split conjunction",                      "level": 4},
    {"premises": ["P -> (Q -> R)", "P", "Q"],            "goal": "R",
     "name": "curried implication",                    "level": 4},
    {"premises": ["(P & Q) -> R", "P", "Q"],             "goal": "R",
     "name": "and-then-impl",                          "level": 4},
    {"premises": ["P | Q", "~P"],                       "goal": "Q",
     "name": "disjunctive syllogism",                  "level": 4},
]
