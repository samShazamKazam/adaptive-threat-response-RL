import numpy as np
import torch

from envs.threat_env import ThreatResponseEnv
from ppo.model import ActorCritic


@torch.no_grad()
def evaluate(model, episodes=50, seed=42):
    env = ThreatResponseEnv(seed=seed)
    returns, resolved, missed = [], 0, 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(obs_t)
            action = int(torch.argmax(logits, dim=-1).item())  # greedy
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        resolved += info["incidents_resolved"]
        missed += info["threats_missed"]
    return np.mean(returns), resolved / episodes, missed / episodes


if __name__ == "__main__":
    env = ThreatResponseEnv()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load("models/threat_ppo_scratch.pt"))
    model.eval()

    mean_ret, avg_resolved, avg_missed = evaluate(model)
    print(f"Mean return: {mean_ret:.2f} | "
          f"resolved/ep: {avg_resolved:.2f} | missed/ep: {avg_missed:.2f}")