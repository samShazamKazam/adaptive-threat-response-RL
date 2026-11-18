from evaluate_all import heuristic_policy, load_trained_policy, run_policy


def test_policy_does_not_regress():
    policy = load_trained_policy("models/threat_ppo_scratch.pt")
    mean_r, std_r, resolved, missed = run_policy(policy, episodes=50, seed=42)

    assert mean_r >= 30.0, f"Mean return regressed: {mean_r:.2f}"
    assert missed <= 1.5, f"Missed threats regressed: {missed:.2f}"
    assert resolved >= 8.5, f"Resolved incidents regressed: {resolved:.2f}"

def test_beats_heuristic():
    ppo_r = run_policy(load_trained_policy("models/threat_ppo_scratch.pt"), episodes=50, seed=42)[0]
    heur_r = run_policy(heuristic_policy, episodes=50, seed=42)[0]
    assert ppo_r > heur_r

    