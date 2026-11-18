# sanity_check.py
import numpy as np
from envs.threat_env import ThreatResponseEnv

env = ThreatResponseEnv(seed=42)
obs, info = env.reset()
print("obs shape:", obs.shape)          # should be (102,)
print("action space:", env.action_space) # Discrete(5)

# Run a random episode
total_reward = 0.0
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Random episode return: {total_reward:.2f}")
print(f"Info: {info}")