"""
Evaluate the trained PPO policy against random and heuristic baselines.

Usage:
    python evaluate_all.py
    python evaluate_all.py --model models/threat_ppo_scratch.pt --episodes 100
"""

import argparse
import numpy as np
import torch

from envs.threat_env import ThreatResponseEnv
from ppo.model import ActorCritic
from collections import Counter


# ----------------------------------------------------------------------
# Policy functions
# ----------------------------------------------------------------------

def random_policy(obs, env):
    """Uniformly random action."""
    return env.action_space.sample()


def heuristic_policy(obs, env):
    """
    Hand-written rule:
      - If the top alert hasn't been investigated, investigate it.
      - If it has been investigated, contain it if revealed as a threat,
        otherwise dismiss it.
      - If the top alert is very severe, contain it immediately.
    """
    if not env.queue:
        return env.WAIT

    # Sort by severity descending so queue[0] is the highest priority
    env.queue.sort(key=lambda a: a["severity"], reverse=True)
    top = env.queue[0]

    if top["investigated"]:
        return env.CONTAIN if top.get("revealed_threat") else env.DISMISS

    if top["severity"] > 0.7:
        return env.CONTAIN

    return env.INVESTIGATE


def load_trained_policy(model_path):
    """Load a trained ActorCritic checkpoint and return a greedy policy fn."""
    env = ThreatResponseEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    @torch.no_grad()
    def policy(obs, _env):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, _ = model(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    return policy


# ----------------------------------------------------------------------
# Evaluation loop
# ----------------------------------------------------------------------

def run_policy(policy_fn, episodes=100, seed=42):
    """
    Run a policy for `episodes` episodes.

    Returns a dict with mean/std return, resolved incidents per episode,
    missed threats per episode, and budget remaining at episode end.
    """
    env = ThreatResponseEnv(seed=seed)
    returns, resolved, missed, budget_left = [], [], [], []
    action_counts = Counter()

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        info = {}

        while not done:
            action = policy_fn(obs, env)
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated

        returns.append(total)
        resolved.append(info.get("incidents_resolved", 0))
        missed.append(info.get("threats_missed", 0))
        budget_left.append(info.get("budget_remaining", 0.0))

    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "resolved_mean": float(np.mean(resolved)),
        "missed_mean": float(np.mean(missed)),
        "budget_left_mean": float(np.mean(budget_left)),
        "action_counts": dict(action_counts),
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/threat_ppo_scratch.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trained = load_trained_policy(args.model)

    policies = [
        ("Random", random_policy),
        ("Heuristic", heuristic_policy),
        ("PPO", trained),
    ]

    results = {}
    print(f"\nEvaluating {args.episodes} episodes per policy (seed={args.seed})\n")
    print(f"{'Policy':>12} | {'Return':>17} | {'Resolved':>9} | "
          f"{'Missed':>7} | {'Budget left':>11}")
    print("-" * 70)


    for seed in [42, 123, 456, 789, 1011]:
        for name, fn in policies[1:]:  # skip random policy for brevity
            r = run_policy(fn, episodes=args.episodes, seed=seed)
            results[name] = r
            action_counts = {
                action_name: r["action_counts"][action]
                for action, action_name in ThreatResponseEnv.ACTION_NAMES.items()
                if action in r["action_counts"]
            }
            print(
                f"{name:>12} | "
                f"{seed:>4d} | "
                f"{r['return_mean']:7.2f} ± {r['return_std']:5.2f} | "
                f"{r['resolved_mean']:9.2f} | "
                f"{r['missed_mean']:7.2f} | "
                f"{r['budget_left_mean']:11.1f}"
                f" | {action_counts}"
            )

    # Improvement over the heuristic baseline
    base = results["Heuristic"]["return_mean"]
    ppo = results["PPO"]["return_mean"]
    if abs(base) > 1e-6:
        improvement = 100.0 * (ppo - base) / abs(base)
        print(f"\nPPO improvement over heuristic: {improvement:.1f}%")

    # Domain-level comparison
    base_missed = results["Heuristic"]["missed_mean"]
    ppo_missed = results["PPO"]["missed_mean"]
    print(f"Missed threats per episode: heuristic {base_missed:.2f} → PPO {ppo_missed:.2f}")

if __name__ == "__main__":
    main()


    