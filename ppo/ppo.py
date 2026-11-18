import torch
import torch.nn as nn


class PPO:
    """
    Proximal Policy Optimization (Schulman et al. 2017).

    Loss = - min( ratio * A,
                  clip(ratio, 1-eps, 1+eps) * A )
           + vf_coef * MSE(V(s), returns)
           - ent_coef * entropy

    The clipping is what keeps the policy from moving too far in one update,
    which is the "proximal" part and the source of PPO's stability.
    """

    def __init__(
        self,
        model: nn.Module,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=lr_critic)

    def update(self, buffer, advantages, returns):
        # Normalize advantages (standard PPO trick: reduces variance)
        adv = torch.as_tensor(advantages, device=self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        advantages_np = adv.cpu().numpy()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_actor_grad_norm = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for obs, actions, old_log_probs, adv_b, ret_b in buffer.get_batches(
                advantages_np, returns, self.batch_size, self.device
            ):
                # ---- Evaluate current policy on the same actions ----
                log_probs, entropy, values = self.model.evaluate_actions(obs, actions)

                # ---- Policy (actor) loss: clipped surrogate ----
                # -- more probable and better than expected -> clipped, to limit the increase in probability
                # -- less probable and better than expected -> not clipped, to increase the probability
                # -- more probable and worse than expected -> not clipped, to decrease the probability
                # -- less probable and worse than expected -> clipped, to limit the decrease in probability
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value (critic) loss ----
                value_loss = nn.functional.huber_loss(values, ret_b)

                # ---- Entropy bonus (encourages exploration) ----
                entropy_loss = -entropy.mean()

                actor_grad_norm = self._actor_grad_norm()
                total_actor_grad_norm += actor_grad_norm

                self.actor_optimizer.zero_grad()
                (policy_loss + self.ent_coef * entropy_loss).backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.vf_coef * value_loss).backward()
                self.critic_optimizer.step()

                nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.max_grad_norm)

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "actor_grad_norm": total_actor_grad_norm / n_updates, 
        }

    def _actor_grad_norm(self) -> float:
        """
        Compute the L2 norm of the actor's gradients.

        Must be called AFTER loss.backward() and BEFORE optimizer.step()
        (and before clip_grad_norm_, if you want the pre-clipping norm).

        Assumes the actor is accessible as self.model.actor when the
        network is split into separate actor and critic trunks. If the
        network is shared, adjust to target the actor head only.
        """
        total_norm = 0.0
        for p in self.model.actor.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().norm(2).item() ** 2
        return total_norm ** 0.5