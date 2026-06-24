"""
HRL training loop.

Outer loop  → manager selects an option (subgoal / strategy)
Inner loop  → worker executes inference steps toward that subgoal

The manager is rewarded when an option terminates successfully.
The worker is rewarded per-step as it moves toward the subgoal.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from core.proof_env import ProofEnv, state_to_obs, CURRICULUM
from core.proof_state import ProofState, LemmaLibrary
from core.inference import ALL_RULES
from agent.worker import WorkerAgent
from agent.manager import ManagerAgent, OptionType, OPTION_NAMES


# ── Training config ───────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    n_episodes: int      = 500
    max_worker_steps: int = 30
    max_options_per_ep: int = 5
    worker_lr: float     = 1e-3
    manager_lr: float    = 5e-4
    use_llm: bool        = False       # set True to enable LLM conjecture
    curriculum_warmup: int = 100       # episodes before using full curriculum
    save_dir: str        = "checkpoints"
    log_every: int       = 25


# ── Episode result ─────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode: int
    theorem_name: str
    proved: bool
    total_reward: float
    n_options: int
    n_worker_steps: int
    manager_options: List[str] = field(default_factory=list)
    lemmas_added: int = 0


# ── Training ──────────────────────────────────────────────────────────────────

def train(config: TrainConfig = TrainConfig()) -> List[EpisodeResult]:
    os.makedirs(config.save_dir, exist_ok=True)

    library = LemmaLibrary()
    lib_path = os.path.join(config.save_dir, "lemma_library.json")
    library.load(lib_path)

    env = ProofEnv(CURRICULUM, library, max_steps=config.max_worker_steps)

    # Determine obs_dim from a reset
    obs, _ = env.reset()
    obs_dim = obs.shape[0]

    worker  = WorkerAgent(obs_dim=obs_dim, n_actions=len(ALL_RULES))
    manager = ManagerAgent(obs_dim=obs_dim, use_llm=config.use_llm)

    # Try loading existing checkpoints
    worker_path  = os.path.join(config.save_dir, "worker.pt")
    manager_path = os.path.join(config.save_dir, "manager.pt")
    if os.path.exists(worker_path):
        worker.load(worker_path)
        print(f"Loaded worker from {worker_path}")
    if os.path.exists(manager_path):
        manager.load(manager_path)
        print(f"Loaded manager from {manager_path}")

    history: List[EpisodeResult] = []

    for ep in range(config.n_episodes):
        library.episode_count = ep
        lib_before = library.size()

        # Curriculum: start with easy theorems, expand over time
        if ep < config.curriculum_warmup:
            level_cap = 2
        elif ep < config.curriculum_warmup * 2:
            level_cap = 3
        else:
            level_cap = 4

        eligible = [i for i, t in enumerate(CURRICULUM) if t.get("level", 1) <= level_cap]
        thm_idx = np.random.choice(eligible)
        thm = CURRICULUM[thm_idx]

        obs, _ = env.reset(theorem_idx=thm_idx)
        state = env.state

        total_reward = 0.0
        ep_proved = False
        option_log = []
        total_worker_steps = 0
        n_options = 0

        manager_obs = obs.copy()

        for opt_step in range(config.max_options_per_ep):
            n_options += 1

            # Manager selects option
            option_type, subgoal, reasoning = manager.select_option(
                manager_obs, state, library
            )
            option_log.append(f"{OPTION_NAMES[option_type]}: {reasoning}")

            # If subgoal is set, redirect the worker env
            if subgoal is not None and subgoal != state.goal:
                env.set_subgoal(subgoal)

            # Worker executes steps toward subgoal
            worker_reward = 0.0
            option_done = False

            for w_step in range(config.max_worker_steps):
                action = worker.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                worker_reward += reward
                total_worker_steps += 1

                worker.push(obs, action, reward, next_obs, terminated or truncated)
                worker.update()

                obs = next_obs
                state = env.state

                if terminated or truncated:
                    option_done = True
                    if info.get("proved"):
                        ep_proved = True
                    break

            # Manager update (option-level reward)
            option_reward = worker_reward
            if option_done and info.get("proved"):
                option_reward += 2.0  # bonus for successful option

            total_reward += option_reward
            next_manager_obs = obs.copy()
            manager.push(
                manager_obs, option_type, option_reward,
                next_manager_obs, ep_proved
            )
            manager.update()
            manager_obs = next_manager_obs

            if ep_proved:
                break

            # If worker timed out without proving, reset for next option attempt
            if not ep_proved:
                obs, _ = env.reset(theorem_idx=thm_idx)
                state = env.state
                manager_obs = obs.copy()

        lemmas_added = library.size() - lib_before

        result = EpisodeResult(
            episode=ep,
            theorem_name=thm["name"],
            proved=ep_proved,
            total_reward=total_reward,
            n_options=n_options,
            n_worker_steps=total_worker_steps,
            manager_options=option_log,
            lemmas_added=lemmas_added,
        )
        history.append(result)

        # Logging
        if ep % config.log_every == 0:
            recent = history[-config.log_every:]
            win_rate = sum(r.proved for r in recent) / len(recent)
            print(
                f"Ep {ep:4d} | "
                f"Theorem: {thm['name']:<30} | "
                f"Proved: {'✓' if ep_proved else '✗'} | "
                f"Win%: {win_rate:.1%} | "
                f"Library: {library.size()} lemmas | "
                f"Reward: {total_reward:.2f}"
            )

        # Save periodically
        if ep % 100 == 0 and ep > 0:
            worker.save(worker_path)
            manager.save(manager_path)
            library.save(lib_path)

    # Final save
    worker.save(worker_path)
    manager.save(manager_path)
    library.save(lib_path)
    print(f"\nTraining complete. Library has {library.size()} lemmas.")

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--llm", action="store_true", help="Enable LLM conjecture")
    args = parser.parse_args()

    cfg = TrainConfig(n_episodes=args.episodes, use_llm=args.llm)
    train(cfg)
