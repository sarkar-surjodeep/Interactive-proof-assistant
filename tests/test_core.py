
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.formula import Atom, And, Or, Implies, Not, Iff, is_tautology, evaluate
from core.parser import parse
from core.inference import ALL_RULES, apply_rule, goal_reached, RULE_NAMES
from core.proof_state import ProofState, LemmaLibrary
from core.proof_env import ProofEnv, state_to_obs, CURRICULUM


def test_formula():
    P, Q, R = Atom("P"), Atom("Q"), Atom("R")
    f = Implies(And(P, Q), R)
    assert f.atoms() == frozenset({"P", "Q", "R"})
    assert f.complexity() == 2  # Implies(And(P,Q), R) = 1 + 1 + 0 = 2
    assert str(f) == "((P ∧ Q) → R)"
    print("✓ formula")


def test_parser():
    cases = [
        ("P", "P"),
        ("~P", "¬P"),
        ("P & Q", "(P ∧ Q)"),
        ("P | Q", "(P ∨ Q)"),
        ("P -> Q", "(P → Q)"),
        ("P <-> Q", "(P ↔ Q)"),
        ("(P -> Q) & (Q -> R)", "((P → Q) ∧ (Q → R))"),
        ("~~P", "¬(¬P)"),
    ]
    for inp, expected in cases:
        result = str(parse(inp))
        assert result == expected, f"parse({inp!r}) = {result!r}, expected {expected!r}"
    print("✓ parser")


def test_tautology():
    assert is_tautology(parse("P -> P"))
    assert is_tautology(parse("P | ~P"))
    assert not is_tautology(parse("P -> Q"))
    assert is_tautology(parse("(P -> Q) -> ((Q -> R) -> (P -> R))"))
    print("✓ tautology checker")


def test_modus_ponens():
    P = parse("P")
    PQ = parse("P -> Q")
    Q = parse("Q")
    premises = frozenset({P, PQ})
    results = apply_rule(0, premises, Q)  # rule 0 = modus_ponens
    assert any(Q in ps for ps in results), f"MP failed, results: {results}"
    print("✓ modus ponens")


def test_hyp_syll():
    PQ = parse("P -> Q")
    QR = parse("Q -> R")
    PR = parse("P -> R")
    premises = frozenset({PQ, QR})
    # hyp_syll is rule 6
    rule_idx = RULE_NAMES.index("hyp_syll")
    results = apply_rule(rule_idx, premises, PR)
    assert any(PR in ps for ps in results), f"Hyp syll failed: {results}"
    print("✓ hypothetical syllogism")


def test_proof_env():
    library = LemmaLibrary()
    env = ProofEnv(CURRICULUM, library, max_steps=30)
    obs, _ = env.reset(theorem_idx=0)  # modus ponens theorem
    assert obs.shape[0] > 0

    # Run a few steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print(f"  Proved in {env.step_count} steps!")
            break
    print("✓ proof env")


def test_state_encoding():
    library = LemmaLibrary()
    env = ProofEnv(CURRICULUM, library)
    obs, _ = env.reset()
    assert obs.dtype.kind == 'f'
    assert (obs >= 0).all() and (obs <= 1).all()
    print(f"✓ state encoding (dim={obs.shape[0]})")


def test_lemma_library():
    lib = LemmaLibrary()
    f = parse("P -> P")
    lib.add(f, ["impl_intro"])
    assert lib.contains(f)
    assert lib.size() == 1
    lib.use(f)
    assert lib.get(f).times_used == 1
    print("✓ lemma library")


if __name__ == "__main__":
    test_formula()
    test_parser()
    test_tautology()
    test_modus_ponens()
    test_hyp_syll()
    test_proof_env()
    test_state_encoding()
    test_lemma_library()
    print("\n✓ All tests passed")
