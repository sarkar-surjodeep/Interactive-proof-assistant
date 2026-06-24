

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import time
import json
from typing import List, Optional

from core.formula import Formula
from core.parser import parse
from core.proof_state import ProofState, LemmaLibrary
from core.proof_env import ProofEnv, state_to_obs, CURRICULUM
from core.inference import ALL_RULES, apply_rule
from agent.worker import WorkerAgent
from agent.manager import ManagerAgent, OptionType, OPTION_NAMES


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HRL Proof Assistant",
    page_icon="∴",
    layout="wide",
)

st.markdown("""
<style>
.step-box {
    background: var(--background-color);
    border-left: 3px solid #7F77DD;
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-family: monospace;
    font-size: 13px;
}
.manager-box {
    background: #f0f4ff;
    border: 1px solid #7F77DD;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
}
.lemma-tag {
    display: inline-block;
    background: #e8f5e9;
    border: 1px solid #66bb6a;
    border-radius: 12px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 12px;
    font-family: monospace;
}
.proved-banner {
    background: #e8f5e9;
    border: 2px solid #43a047;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    color: #2e7d32;
}
.failed-banner {
    background: #fff3e0;
    border: 2px solid #ff9800;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
    font-size: 16px;
    color: #e65100;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────

def init_state():
    if "library" not in st.session_state:
        st.session_state.library = LemmaLibrary()
        lib_path = "checkpoints/lemma_library.json"
        if os.path.exists(lib_path):
            st.session_state.library.load(lib_path)

    if "worker" not in st.session_state:
        # We'll init after we know obs_dim
        st.session_state.worker = None
        st.session_state.manager = None

    if "proof_log" not in st.session_state:
        st.session_state.proof_log = []

    if "win_history" not in st.session_state:
        st.session_state.win_history = []

    if "episode_count" not in st.session_state:
        st.session_state.episode_count = 0


def get_agents(obs_dim: int):
    if st.session_state.worker is None:
        st.session_state.worker  = WorkerAgent(obs_dim=obs_dim, n_actions=len(ALL_RULES))
        st.session_state.manager = ManagerAgent(obs_dim=obs_dim, use_llm=False)
        # Load if available
        if os.path.exists("checkpoints/worker.pt"):
            try:
                st.session_state.worker.load("checkpoints/worker.pt")
            except Exception:
                pass
        if os.path.exists("checkpoints/manager.pt"):
            try:
                st.session_state.manager.load("checkpoints/manager.pt")
            except Exception:
                pass
    return st.session_state.worker, st.session_state.manager


# ── Proof runner ──────────────────────────────────────────────────────────────

def run_proof_interactive(premises_strs: List[str], goal_str: str, animate: bool = True):
    """Run one proof episode and stream the results to the UI."""
    library = st.session_state.library

    theorem = {"premises": premises_strs, "goal": goal_str, "name": goal_str}
    env = ProofEnv([theorem], library, max_steps=30)
    obs, _ = env.reset(theorem_idx=0)
    obs_dim = obs.shape[0]

    worker, manager = get_agents(obs_dim)

    log = []
    proved = False
    all_options = []

    for opt_step in range(5):
        state = env.state

        # Manager picks option
        option_type, subgoal, reasoning = manager.select_option(
            obs, state, library, greedy=True
        )
        all_options.append((option_type, subgoal, reasoning))
        log.append(("manager", f"[Option {opt_step+1}] {reasoning}"))

        if subgoal is not None and subgoal != state.goal:
            env.set_subgoal(subgoal)

        for w_step in range(30):
            action = worker.select_action(obs, greedy=True)
            rule = ALL_RULES[action]
            obs, reward, terminated, truncated, info = env.step(action)

            if info["fired"]:
                step_desc = f"  {rule.name}: {rule.description}"
                log.append(("worker", step_desc))

            if terminated:
                if info["proved"]:
                    proved = True
                    log.append(("proved", f"✓ Proved: {env.state.goal}"))
                break
            if truncated:
                break

        if proved:
            break

        # Reset env for next option attempt
        obs, _ = env.reset(theorem_idx=0)

    if not proved:
        log.append(("failed", "✗ Could not prove within budget"))

    return proved, log, library


# ── UI ─────────────────────────────────────────────────────────────────────────

def render_log(log_entries):
    for kind, text in log_entries:
        if kind == "manager":
            st.markdown(f'<div class="manager-box">🧠 <b>Manager</b>: {text}</div>', unsafe_allow_html=True)
        elif kind == "worker":
            st.markdown(f'<div class="step-box">⚙️ {text}</div>', unsafe_allow_html=True)
        elif kind == "proved":
            st.markdown(f'<div class="proved-banner">{text}</div>', unsafe_allow_html=True)
        elif kind == "failed":
            st.markdown(f'<div class="failed-banner">{text}</div>', unsafe_allow_html=True)


def render_library(library: LemmaLibrary):
    st.subheader(f"Lemma library ({library.size()} lemmas)")
    if library.size() == 0:
        st.caption("Empty — lemmas appear here as the agent proves them.")
    else:
        for entry in library.recent(20):
            used_text = f"used {entry.times_used}×" if entry.times_used else "new"
            st.markdown(
                f'<span class="lemma-tag">{entry.formula}</span>'
                f'<span style="font-size:11px;color:#888;margin-left:4px">{used_text}</span>',
                unsafe_allow_html=True
            )
            st.markdown("")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_state()

    st.title("∴ HRL Proof Assistant")
    st.caption(
        "Hierarchical Reinforcement Learning for propositional logic theorem proving. "
        "Manager selects proof strategies; Worker applies inference rules."
    )

    col_left, col_right = st.columns([3, 1])

    with col_right:
        render_library(st.session_state.library)

    with col_left:
        # ── Input section ──────────────────────────────────────────────────────
        st.subheader("Theorem to prove")

        mode = st.radio("Input mode", ["Curriculum theorem", "Custom"], horizontal=True)

        if mode == "Curriculum theorem":
            thm_names = [f"[L{t.get('level',1)}] {t['name']}" for t in CURRICULUM]
            choice = st.selectbox("Select theorem", thm_names)
            idx = thm_names.index(choice)
            thm = CURRICULUM[idx]
            premises_strs = thm["premises"]
            goal_str = thm["goal"]

            st.markdown("**Premises:**")
            for p in premises_strs:
                st.code(p, language=None)
            st.markdown(f"**Goal:** `{goal_str}`")

        else:
            st.caption(
                "Syntax: `P`, `Q`, `R` for atoms · `->` implication · "
                "`&` and · `|` or · `~` not · `<->` iff · parentheses for grouping"
            )
            raw_premises = st.text_area(
                "Premises (one per line)",
                value="P -> Q\nQ -> R",
                height=100,
            )
            goal_str = st.text_input("Goal", value="P -> R")
            premises_strs = [p.strip() for p in raw_premises.strip().split("\n") if p.strip()]

            # Validate
            valid = True
            for p in premises_strs:
                try:
                    parse(p)
                except Exception as e:
                    st.error(f"Parse error in premise `{p}`: {e}")
                    valid = False
            try:
                parse(goal_str)
            except Exception as e:
                st.error(f"Parse error in goal: {e}")
                valid = False

            if not valid:
                return

        # ── Run proof ──────────────────────────────────────────────────────────
        if st.button("▶ Run proof", type="primary"):
            st.session_state.proof_log = []
            st.session_state.episode_count += 1

            with st.spinner("Searching for proof..."):
                proved, log, library = run_proof_interactive(premises_strs, goal_str)
                st.session_state.proof_log = log
                st.session_state.library = library
                st.session_state.win_history.append(1 if proved else 0)

            st.rerun()

        # ── Show log ───────────────────────────────────────────────────────────
        if st.session_state.proof_log:
            st.divider()
            st.subheader("Proof trace")
            render_log(st.session_state.proof_log)

        # ── Win rate chart ─────────────────────────────────────────────────────
        if len(st.session_state.win_history) > 1:
            st.divider()
            st.subheader("Proof success rate")
            wins = st.session_state.win_history
            # Rolling 10-episode average
            window = 10
            rolling = [
                np.mean(wins[max(0, i-window):i+1])
                for i in range(len(wins))
            ]
            st.line_chart({"Win rate (rolling 10)": rolling})

        # ── Sidebar training controls ──────────────────────────────────────────
        with st.sidebar:
            st.header("Training")
            st.caption("Run background training to improve the agent's proof search.")

            n_ep = st.number_input("Training episodes", min_value=10, max_value=1000, value=100, step=10)

            if st.button("🏋 Train agent"):
                progress = st.progress(0)
                status = st.empty()

                from train import train, TrainConfig
                cfg = TrainConfig(
                    n_episodes=int(n_ep),
                    use_llm=False,
                    log_every=max(1, int(n_ep) // 10),
                )

                results = []
                # We run in-process (blocking for demo)
                import threading

                def _train():
                    r = train(cfg)
                    results.extend(r)

                t = threading.Thread(target=_train, daemon=True)
                t.start()

                ep = 0
                while t.is_alive():
                    time.sleep(0.5)

                t.join()

                # Reload library
                if os.path.exists("checkpoints/lemma_library.json"):
                    st.session_state.library.load("checkpoints/lemma_library.json")

                # Reset agents so they reload checkpoints
                st.session_state.worker  = None
                st.session_state.manager = None

                st.success(f"Training done! Library: {st.session_state.library.size()} lemmas")
                st.rerun()

            st.divider()
            st.header("Inference rules")
            for rule in ALL_RULES:
                st.markdown(f"**{rule.name}**  \n`{rule.description}`")

            st.divider()
            st.header("About")
            st.markdown("""
This demo implements the **Options Framework** for HRL:

- **Manager** selects a proof strategy (option) — direct, contradiction, case split, library reuse, or LLM conjecture
- **Worker** applies inference rules step-by-step toward the subgoal
- **Lemma library** grows as proofs succeed and is reused in future episodes

""")


if __name__ == "__main__":
    main()
