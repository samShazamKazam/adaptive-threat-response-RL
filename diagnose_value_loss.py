# diagnose_value_loss.py
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# 1. Collect a rollout with the current model, then inspect the returns
from envs.threat_env import ThreatResponseEnv

env = ThreatResponseEnv(seed=42)
returns = []
episode_return = 0.0

obs, _ = env.reset()
for _ in range(5000):
    action = env.action_space.sample()   # random policy, just to measure return distribution
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward
    if terminated or truncated:
        returns.append(episode_return)
        episode_return = 0.0
        obs, _ = env.reset()

returns = np.array(returns)
print(f"Return mean: {returns.mean():.2f}")
print(f"Return std:  {returns.std():.2f}")
print(f"Return min:  {returns.min():.2f}")
print(f"Return max:  {returns.max():.2f}")
print(f"Return variance (theoretical floor for MSE if predicting the mean): {returns.var():.2f}")
print(f"Value loss plateau ≈ {returns.var():.2f} would mean 'predicting the mean'")

# If your value loss ≈ returns.var(), the critic is predicting the mean.
