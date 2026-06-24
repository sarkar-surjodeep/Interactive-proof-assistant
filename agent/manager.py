

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Optional, List, Dict, Any, Tuple
from enum import IntEnum

from core.formula import Formula, Atom, Not, Implies, And
from core.proof_state import ProofState, LemmaLibrary
from core.parser import parse


# ── Option types ──────────────────────────────────────────────────────────────

class OptionType(IntEnum):
    DIRECT        = 0
    CONTRADICTION = 1
    CASE_SPLIT    = 2
    LIBRARY_REUSE = 3
    LLM_CONJECTURE = 4


OPTION_NAMES = {
    OptionType.DIRECT:         "Direct proof",
    OptionType.CONTRADICTION:  "Proof by contradiction",
    OptionType.CASE_SPLIT:     "Case split",
    OptionType.LIBRARY_REUSE:  "Library lemma reuse",
    OptionType.LLM_CONJECTURE: "LLM conjecture",
}


# ── Manager network ───────────────────────────────────────────────────────────

class ManagerNet(nn.Module):
    def __init__(self, obs_dim: int, n_options: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_options),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Option result ─────────────────────────────────────────────────────────────

class OptionResult:
    """What the manager gets back after a worker episode."""
    def __init__(
        self,
        option_type: OptionType,
        subgoal: Optional[Formula],
        proved: bool,
        steps: List[str],
        reward: float,
    ):
        self.option_type = option_type
        self.subgoal = subgoal
        self.proved = proved
        self.steps = steps
        self.reward = reward


# ── Manager agent ─────────────────────────────────────────────────────────────

class ManagerAgent:
   

    N_OPTIONS = len(OptionType)

    def __init__(
        self,
        obs_dim: int,
        lr: float = 5e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 500,
        use_llm: bool = True,
    ):
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.use_llm = use_llm

        self.policy_net = ManagerNet(obs_dim, self.N_OPTIONS)
        self.target_net = ManagerNet(obs_dim, self.N_OPTIONS)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=2000)
        self.batch_size = 32
        self.target_update_freq = 20
        self.loss_history: List[float] = []

        # LLM client (lazy-loaded)
        self._llm_client = None

    # ── Action selection ──────────────────────────────────────────────────────

    def select_option(
        self,
        obs: np.ndarray,
        state: ProofState,
        library: LemmaLibrary,
        greedy: bool = False,
    ) -> Tuple[OptionType, Optional[Formula], str]:
        
        eps = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
              np.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        # Mask options that aren't applicable
        available = self._available_options(state, library)

        if not greedy and random.random() < eps:
            option_type = random.choice(available)
        else:
            with torch.no_grad():
                q = self.policy_net(torch.FloatTensor(obs).unsqueeze(0))[0]
                # mask unavailable
                mask = torch.full((self.N_OPTIONS,), -1e9)
                for o in available:
                    mask[int(o)] = q[int(o)]
                option_type = OptionType(int(mask.argmax().item()))

        subgoal, reasoning = self._resolve_option(option_type, state, library)
        return option_type, subgoal, reasoning

    def _available_options(self, state: ProofState, library: LemmaLibrary) -> List[OptionType]:
        available = [OptionType.DIRECT, OptionType.CONTRADICTION]
        if state.goal.atoms():
            available.append(OptionType.CASE_SPLIT)
        if library.size() > 0:
            available.append(OptionType.LIBRARY_REUSE)
        if self.use_llm:
            available.append(OptionType.LLM_CONJECTURE)
        return available

    def _resolve_option(
        self,
        option_type: OptionType,
        state: ProofState,
        library: LemmaLibrary,
    ) -> Tuple[Optional[Formula], str]:
        

        if option_type == OptionType.DIRECT:
            return None, f"Attempting direct proof of {state.goal}"

        if option_type == OptionType.CONTRADICTION:
            from core.formula import Not
            neg_goal = Not(state.goal)
            return neg_goal, f"Assuming ¬({state.goal}) to derive contradiction"

        if option_type == OptionType.CASE_SPLIT:
            # Pick first atom in goal to split on
            atoms = sorted(state.goal.atoms())
            if atoms:
                a = Atom(atoms[0])
                subgoal = Implies(a, state.goal)
                return subgoal, f"Case split on {a}: proving {a} → {state.goal}"
            return None, "Case split failed (no atoms)"

        if option_type == OptionType.LIBRARY_REUSE:
            # Pick the most-used lemma that's relevant to current goal
            relevant = [
                e for e in library.all_lemmas()
                if e.formula.atoms() & state.goal.atoms()
            ]
            if relevant:
                best = max(relevant, key=lambda e: e.times_used)
                library.use(best.formula)
                return best.formula, f"Reusing library lemma: {best.formula}"
            # Fall back to most recently proved
            recent = library.recent(1)
            if recent:
                library.use(recent[0].formula)
                return recent[0].formula, f"Reusing recent lemma: {recent[0].formula}"
            return None, "Library is empty, falling back to direct"

        if option_type == OptionType.LLM_CONJECTURE:
            subgoal, reasoning = self._llm_propose(state, library)
            return subgoal, reasoning

        return None, "Unknown option"

    # ── LLM conjecture ────────────────────────────────────────────────────────

    def _get_llm_client(self):
        if self._llm_client is None:
            import anthropic
            self._llm_client = anthropic.Anthropic()
        return self._llm_client

    def _llm_propose(
        self,
        state: ProofState,
        library: LemmaLibrary,
    ) -> Tuple[Optional[Formula], str]:
        """
        Ask Claude to propose a useful intermediate lemma.
        Falls back to direct proof on any failure.
        """
        premise_strs = [str(p) for p in sorted(str(x) for x in state.premises)]
        prompt = f"""You are helping prove a propositional logic theorem.

Current premises:
{chr(10).join('  ' + p for p in premise_strs[:10])}

Goal to prove: {state.goal}

Known lemmas already proved:
{library.to_context_string()}

Suggest ONE useful intermediate lemma (a formula) that would help prove the goal.
The lemma should be simpler than the goal and derivable from the premises.

Respond with ONLY the formula using this syntax:
- Use P, Q, R, S for atoms
- Use -> for implication
- Use & for conjunction
- Use | for disjunction
- Use ~ for negation
- Use parentheses for grouping

Example valid responses:
P -> R
(P & Q) -> S
~Q -> ~P

Respond with the formula only, nothing else."""

        try:
            client = self._get_llm_client()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = msg.content[0].text.strip()
            # Clean up common LLM output issues
            raw = raw.split("\n")[0].strip().strip("`").strip()
            formula = parse(raw)
            reasoning = f"LLM proposed lemma: {formula}"
            return formula, reasoning
        except Exception as e:
            # Graceful fallback — don't crash the training loop
            return None, f"LLM conjecture failed ({e}), falling back to direct"

    # ── Learning ──────────────────────────────────────────────────────────────

    def push(self, obs, option, reward, next_obs, done):
        self.buffer.append((obs, int(option), reward, next_obs, done))

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        batch = random.sample(self.buffer, self.batch_size)
        obs, opts, rews, next_obs, dones = zip(*batch)

        obs_t      = torch.FloatTensor(np.array(obs))
        opts_t     = torch.LongTensor(opts)
        rews_t     = torch.FloatTensor(rews)
        next_obs_t = torch.FloatTensor(np.array(next_obs))
        dones_t    = torch.FloatTensor(dones)

        q_vals = self.policy_net(obs_t).gather(1, opts_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_obs_t).max(1)[0]
            target = rews_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.functional.smooth_l1_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.steps_done = ckpt["steps_done"]
