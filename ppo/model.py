import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    actor-critic network for the PPO algorithm.

    - the trunk learns a representation of the state.
    - the actor head outputs actions logits -> Categorical distribution.
    - the critic head outputs a scalar state value estimate V(s).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        # Independent actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        # Independent critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

        # Orthogonal init 
        self._init_weights()

    def _init_weights(self):
        for network, output_gain in ((self.actor, 0.01), (self.critic, 1.0)):
            linear_layers = [layer for layer in network if isinstance(layer, nn.Linear)]
            for layer in linear_layers[:-1]:
                nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
                nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)
            nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)  # shape: (batch,)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """Sample an action for a single observation (or batch)."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Used during the update phase: get log-probs and entropy for taken actions."""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value