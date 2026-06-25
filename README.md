# Hierarchical Reinforcement Learning Proof Assistant

A small system that learns to prove logic theorems by thinking in two layers, the way a person does: first decide on a strategy, then carry out the individual steps. It explores a question at the heart of automated reasoning: can hierarchical reinforcement learning help a machine prove theorems and come up with useful lemmas on its own?

It works on propositional logic (the logic of "and", "or", "not", "implies") rather than a full proof assistant like Lean or Coq. That keeps it small enough to read in an afternoon while still showing the core ideas in action.


## Motivation

When a mathematician proves something hard, they don't think about it one tiny inference at a time. They think "I'll prove this by contradiction" or "I'll need a helper lemma about X first," and only then work out the small steps. There are two levels of thinking going on: a planning level and a doing level.

That two-level structure is exactly what hierarchical reinforcement learning (HRL) is built for, and it's why proof search is such a natural fit. A flat agent that picks one inference rule at a time gets lost, because the reward (a finished proof) only arrives at the very end, after a long chain of choices. HRL breaks that long chain into chunks so the agent can learn from each chunk.

So this system has two agents:

**The Manager** is the planner. It looks at the current state of the proof and picks a strategy: try to prove the goal directly, try contradiction, split into cases, reuse a lemma it already knows, or ask a language model to suggest a brand-new helper lemma. It doesn't touch the logic itself, it just sets the direction.

**The Worker** is the doer. It takes whatever subgoal the Manager handed it and chips away at it one inference rule at a time (modus ponens, and-elimination, and so on) until it either reaches the subgoal or runs out of steps.

Every time the Worker proves a lemma, that lemma gets saved in a **library**. In later proofs, the saved lemmas are available to be reused. Over many attempts the Manager starts to learn which lemmas are worth proving and reusing, which is where the "generating useful lemmas" behaviour comes from. This reuse-and-grow idea is borrowed directly from LEGO-Prover (Wang et al., 2023), which does the same thing at a much larger scale on real olympiad problems.


## How a single proof runs

1. You give it a theorem: some premises and a goal to prove.
2. The Manager looks at the situation and picks a strategy. If that strategy is "prove a helper lemma first," the lemma becomes the Worker's new target.
3. The Worker applies inference rules step by step toward that target.
4. If the Worker succeeds, the lemma is added to the library and the Manager gets credit. If it fails or times out, the Manager picks a different strategy and tries again.
5. This repeats until the original goal is proved or the attempt budget runs out.

The rewards are deliberately simple. The Worker gets a small reward each time it makes real progress, a bigger one when it finishes a subgoal, and a large one when the whole theorem is done. The Manager is rewarded when the strategy it chose actually pays off.


## Architecture

**The logic engine.** Formulas are represented as a small tree structure (an atom like P, or a connective like "P implies Q" wrapping smaller formulas). There's a parser so you can type `(P -> Q) & (Q -> R)` instead of building those trees by hand. There are eleven inference rules, each written as a plain function that takes what you currently know and returns what you can newly derive.

**The environment.** This wraps the logic engine in the standard Gymnasium interface used for reinforcement learning. It turns a proof state into a fixed-size vector of numbers the agents can read, and it hands out the rewards described above.

**The two agents.** Both are small neural networks trained with DQN, a standard value-based RL algorithm. They're intentionally tiny, because propositional logic has a small enough search space that nothing bigger is needed.

**The curriculum.** Twenty theorems arranged from easy to hard, in four levels. Level 1 is a single rule (basic modus ponens). Level 4 needs the agent to decompose the problem into several lemmas. Training starts on the easy ones and widens to the harder ones as the agent improves, which is a standard curriculum-learning setup.


## The eleven inference rules

| Rule | What it does |
|------|--------------|
| Modus Ponens | from A and "A implies B", get B |
| Modus Tollens | from "not B" and "A implies B", get "not A" |
| And Introduction | from A and B, get "A and B" |
| And Elimination | from "A and B", get A and get B |
| Or Introduction | from A, get "A or B" |
| Or Elimination | from "A or B", "A implies C", "B implies C", get C |
| Hypothetical Syllogism | from "A implies B" and "B implies C", get "A implies C" |
| Double Negation | from "not not A", get A |
| Contradiction | from A and "not A", anything follows |
| Implication Introduction | assume A, derive B, conclude "A implies B" |
| Iff Elimination | from "A iff B", get both directions of the implication |


## Quick Install

Install the dependencies:

```bash
pip install gymnasium numpy torch anthropic streamlit
```

Train the agent (this creates a `checkpoints/` folder and takes under a minute):

```bash
python train.py --episodes 500
```

Launch the demo, which opens in your browser:

```bash
streamlit run app.py
```

In the demo you can pick a theorem or type your own, watch the Manager choose strategies and the Worker grind through steps, and see the lemma library fill up as you go.

If you want the Manager to use a language model to invent new lemmas, set your Anthropic API key and add the `--llm` flag:

```bash
export ANTHROPIC_API_KEY=your_key_here     # use "set" instead of "export" on Windows
python train.py --episodes 500 --llm
```

To check everything is wired up correctly:

```bash
python tests/test_core.py
```


## Design rationale

The system is built around a question that keeps coming up in automated reasoning: can HRL automate proofs and generate interesting conjectures and lemmas? Each design choice maps onto one piece of that question.

- **Decomposing a proof into subgoals.** The Manager setting lemma subgoals mirrors how a real prover would break a hard theorem into pieces. This subgoal-then-prove structure is the same one used by Draft, Sketch, and Prove (Jiang et al., 2023), where an informal sketch is turned into formal sub-problems.
- **Working at two timescales.** The Manager plans over strategies while the Worker acts over individual steps. This is the options framework from Sutton, Precup, and Singh (1999), the foundational formulation of temporal abstraction in RL, and it's what lets the agent cope with the sparse, far-off reward that defeats a flat agent.
- **A library that grows.** Saving and reusing lemmas is the LEGO-Prover idea (Wang et al., 2023). It's also what makes lemma generation more than a gimmick: a lemma is "useful" if it gets reused, which gives a concrete, measurable signal.

It's just as important to be clear about what this is *not*, because that's where the interesting problems actually live:

- It runs on a toy logic, not a real proof assistant. A serious version would sit on top of Lean 4, Coq, or Isabelle, where the proof states are vastly richer.
- The Worker learns from scratch. A real system would start from a pretrained tactic model rather than a small network, the way HyperTree Proof Search (Lample et al., 2022) pairs a transformer with its search.
- The hardest open problem is barely touched here: what makes a conjecture *interesting*? Rewarding a lemma for being provable and for being reused is tractable, but mathematical interestingness is far subtler, and an agent rewarded only for reuse will happily churn out true-but-boring lemmas. Pinning down a better signal is one of the genuinely hard parts of this whole area.


## Project layout

```
hrl_prover/
├── core/
│   ├── formula.py       formula trees and a truth-table checker
│   ├── parser.py        turns text like "P -> Q" into formula trees
│   ├── inference.py     the eleven inference rules
│   ├── proof_state.py   the proof state and the lemma library
│   └── proof_env.py     the Gymnasium environment and the 20-theorem curriculum
├── agent/
│   ├── worker.py        the step-level DQN agent
│   └── manager.py       the strategy-level agent and the LLM lemma proposer
├── train.py             the training loop that ties Manager and Worker together
├── app.py               the Streamlit demo
└── tests/
    └── test_core.py     unit tests for the logic engine
```

## References

[1] Sutton, Richard S., Doina Precup, and Satinder Singh. "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." Artificial intelligence 112.1-2 (1999): 181-211.

[2] Lample, Guillaume, et al. "Hypertree proof search for neural theorem proving." Advances in neural information processing systems 35 (2022): 26337-26349.

[3] Jiang, Albert Q., et al. "Draft, sketch, and prove: Guiding formal theorem provers with informal proofs." arXiv preprint arXiv:2210.12283 (2022).

[4] Wang, Haiming, et al. "Lego-prover: Neural theorem proving with growing libraries." International Conference on Learning Representations. Vol. 2024. 2024.
