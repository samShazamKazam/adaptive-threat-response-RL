import json
import numpy as np
import torch
from collections import deque
from pathlib import Path

from envs.threat_env import ThreatResponseEnv
from ppo.model import ActorCritic
from ppo.rollout import RolloutBuffer
from ppo.ppo import PPO
from torch.utils.tensorboard import SummaryWriter


def train(
    total_timesteps: int = 200_000,
    horizon: int = 2048,       # max steps collected per update
    gamma: float = 0.99,
    lam: float = 0.95,
    seed: int = 0,
    max_steps: int = 50,
    max_queue: int = 20,
    budget: float = 100.0,

):

    with Path("ppo/config.json").open(encoding="utf-8") as config_file:
        ppo_config = json.load(config_file)

    writer = SummaryWriter("tb_logs/threat_ppo")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = ThreatResponseEnv(max_steps=max_steps, max_queue=max_queue, budget=budget, seed=seed)
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = ppo_config["device"]

    model = ActorCritic(obs_dim, n_actions)
    ppo = PPO(
        model,
        lr_actor=ppo_config["lr_actor"],
        lr_critic=ppo_config["lr_critic"],
        clip_eps=ppo_config["clip_eps"],
        vf_coef=ppo_config["vf_coef"],
        ent_coef=ppo_config["ent_coef"],
        max_grad_norm=ppo_config["max_grad_norm"],
        n_epochs=ppo_config["n_epochs"],
        batch_size=ppo_config["batch_size"],
        device=device,
    )
    buffer = RolloutBuffer(horizon, obs_dim, gamma=gamma, lam=lam)

    episode_returns = deque(maxlen=20)
    current_return = 0.0
    global_step = 0
    episode_count = 0

    while global_step < total_timesteps:
        buffer.reset()
        obs, _ = env.reset()

        for _ in range(horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, log_prob, value = model.act(obs_t)

            a = int(action.item())
            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action=a,
                log_prob=float(log_prob.item()),
                reward=reward,
                value=float(value.item()),
                done=done,
            )

            current_return += reward
            obs = next_obs
            global_step += 1

            if done:
                episode_returns.append(current_return)
                current_return = 0.0
                episode_count += 1
                obs, _ = env.reset()

            if global_step >= total_timesteps:
                break

        # Bootstrap with the value of the final state
        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value = model(last_obs_t)
            last_value = float(last_value.item())

        advantages, returns = buffer.compute_gae(last_value)
        stats = ppo.update(buffer, advantages, returns)

        mean_return = np.mean(episode_returns) if episode_returns else float("nan")

        writer.add_scalar("train/mean_return", mean_return, global_step)
        writer.add_scalar("train/policy_loss", stats["policy_loss"], global_step)
        writer.add_scalar("train/value_loss", stats["value_loss"], global_step)
        writer.add_scalar("train/entropy", stats["entropy"], global_step)
        writer.add_scalar("train/episodes", episode_count, global_step)
        writer.add_scalar("train/actor_grad_norm", stats["actor_grad_norm"], global_step) 

        print(
            f"step {global_step:7d} | episodes {episode_count:5d} | "
            f"mean return (last 20) {mean_return:8.2f} | "
            f"pi_loss {stats['policy_loss']:7.3f} | "
            f"v_loss {stats['value_loss']:8.3f} | "
            f"entropy {stats['entropy']:.3f} | "
            f"actor_grad_norm {stats['actor_grad_norm']:.4f}"
        )

    writer.close()
    torch.save(model.state_dict(), "models/threat_ppo_scratch.pt")
    print("Saved model to models/threat_ppo_scratch.pt")
    return model


if __name__ == "__main__":
    config_path = Path("config.json")
    config = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)

    train(**{
        key: config[key]
        for key in (
            "total_timesteps", "horizon", "gamma", "lam", "seed", "max_steps", "max_queue", "budget"
        )
        if key in config
    })