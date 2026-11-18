# Adaptive Threat Response RL

A reinforcement learning environment that models Security Operations Center (SOC) alert triage as a sequential decision problem, plus a from-scratch PyTorch implementation of Proximal Policy Optimization (PPO).

The agent learns to triage a stream of security alerts—investigating, containing, escalating, dismissing, or waiting—under a finite budget, balancing threat containment against wasted effort.

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Environment Design](#environment-design)
- [PPO Implementation](#ppo-implementation)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Regression Testing](#regression-testing)
- [Results](#results)
- [Design Tradeoffs](#design-tradeoffs)
- [What I Learned](#what-i-learned)
- [Extensions](#extensions)
- [References](#references)

---

## Overview

Most reinforcement learning portfolio projects are either games (CartPole, Atari) or robotics (MuJoCo). This project takes a different approach: it models a **real operational problem**—SOC alert triage—as a Markov Decision Process, and implements PPO from scratch so every part of the algorithm is visible and explainable. 

**What makes this project different:**

- **Custom environment** — not a library environment; the MDP is designed from first principles
- **Domain-grounded** — models a real problem in security operations, fraud detection, and incident response
- **From-scratch PPO** — the clipped objective, GAE, and advantage normalization are all implemented explicitly
- **Production-minded** — includes observability, baseline comparison, and CI regression tests
- **Rigorous evaluation** — compared against random and heuristic baselines, not just "it trained"

---

## The Problem

A Security Operations Center analyst receives a continuous stream of alerts. Each alert may be a **real threat** or a **false positive**. The analyst must decide what to do with each one:

- **Investigate** — gather evidence (costs time, may reveal the truth)
- **Contain** — act immediately (expensive, risky if benign)
- **Escalate** — hand off to a human (moderate cost)
- **Dismiss** — mark as false positive (cheap, catastrophic if wrong)
- **Wait** — let more context arrive (cheap, but the queue grows)

The challenge is that **the analyst doesn't know which alerts are real threats**. Investigation is imperfect. And resources are finite—every action costs budget.

This is a genuine **sequential decision problem**: each action affects the state the agent sees next, and the optimal strategy requires weighing immediate action against future information.

---

## Environment Design

This is a Markov Decision Process (MDP) problem describing a stochastic dynamical system. Note that some properties are hidden from the agent, used by the system in reward calculation  (i.e. alert["is_threat"]). The agent will face 20 alerts per episode, each has six properties.

### Observation Space (102 dimensions)

20 alerts × 6 features = 120

budget remaining (normalized) = 1

step fraction = 1

So total is 120 + 1 + 1 = 122 dimensions

Each alert has 6 features:

| Feature | Range | Meaning |
| :--- | :--- | :--- |
| `severity` | [0, 1] | How severe the alert appears |
| `source_id` | {0,1,2,3} | EDR, SIEM, IAM, or Email |
| `asset_criticality` | [0, 1] | How important the affected asset is |
| `age` | [0, 1] | Normalized steps since arrival |
| `investigated` | {0, 1} | Whether the alert has been investigated |
| `revealed_threat` | {0, 1} | Whether it's a threat |

### Action Space (5 discrete actions)

| Index | Action | Cost | Effect |
| :--- | :--- | :--- | :--- |
| 0 | Investigate | 2.0 | Reveals truth with 90% accuracy |
| 1 | Contain | 5.0 | +10 if threat, −5 if benign |
| 2 | Escalate | 3.0 | +5 if threat, −1 if benign |
| 3 | Dismiss | 0.5 | +1 if benign, −10 if threat |
| 4 | Wait | 1.0 | Time passes; small chance severity updates |

### Reward Structure

| Event | Reward |
| :--- | :--- |
| Contain a real threat | +10.0 |
| Escalate a real threat | +5.0 |
| Correctly dismiss a benign alert | +1.0 |
| First investigation | 0.0 |
| Repeated investigation of an already-investigated alert | −500.0 |
| Contain a benign alert | −5.0 |
| Escalate a benign alert | −1.0 |
| Dismiss a real threat | −10.0 |
| Budget below 30% remaining | Additional penalty increases gradually as the budget approaches zero |
| Unresolved threats at episode end | −2.0 each |

Reaching zero budget terminates the episode; there is no separate fixed −20.0 budget-exhaustion reward penalty.

### Budget

Episodes start with a budget of **100.0**. Each action deducts its cost. As the agent is getting close to depleting its budget, a soft penalty applies, adding to the pressure to maximize reward using the remaining budget.

The budget is calibrated so a straightforward strategy (investigate, then contain) uses roughly 60% of it. A sloppy policy runs out while a good policy succeeds with budget to spare.

### Design Rationale

| Choice | Why |
| :--- | :--- |
| **Discrete actions** | Triage is naturally categorical; no continuous control needed |
| **Deterministic reward** | I put uncertainty in the state (the agent doesn't know which alerts are threats) rather than in the reward (containing a threat always gives the same reward). This means the reward is a deterministic function of the action and the true state, so reward variance comes only from the agent's choices and not from random reward noise. That keeps the gradient signal clean and makes the learning dynamics easier to analyze.|
| **10% investigation error** | Models imperfect information |
| **Step fraction in observation** | normalized measure of how far through the episode the agent is. It matters because the optimal action depends on time remaining: early in the episode, there's time to wait and investigate; late, the agent should act decisively. It gives the agent a sense of urgency. It has to reside in the state as that satisfies the Markov property that is all the info the agent needs must reside the current state; |
| **Budget constraint** | Forces genuine tradeoffs between speed and thoroughness |

---

## PPO Implementation

PPO is implemented from scratch in PyTorch so every component is visible and explainable.

### Key Components

| File | Purpose |
| :--- | :--- |
| `ppo/model.py` | Shared-trunk actor-critic network with orthogonal initialization |
| `ppo/rollout.py` | Rollout buffer with GAE (Generalized Advantage Estimation) |
| `ppo/ppo.py` | Clipped surrogate objective, value loss, entropy bonus |

### The Network

Observation (102)

```
   │
┌──▼──┐
│Trunk│ Linear(102, 128) → Tanh → Linear(128, 128) → Tanh
└──┬──┘
   │
┌──┴──────────────┐
▼                 ▼
Linear(128, 5)    Linear(128, 1)
(logits)          (state value)
```
I am using Tanh since it's zero-centered with output range (-1, 1) and we only have two NN layers so no concern of the gradient is vanishing. 

**Initialization gains:**

The gain is a scaling factor applied after orthogonalization. It controls the magnitude of the layer's output.

For the actor head, the output goes through a softmax (via Categorical(logits=logits)) to produce action probabilities. If the weights are large, the logits are large, and the softmax becomes very peaked—the policy is nearly deterministic from the start.

| Layer | Gain | Why |
| :--- | :--- | :--- |
| Actor | 0.01 | Small logits → near-uniform policy → good exploration , small gain to start with low confident policy|
| Critic | 1.0 | Must output values across the full range of returns |

### The Objective

I am using PPO clipping.

```python
ratio = exp(log_probs_new - log_probs_old)
surr1 = ratio * advantages
surr2 = clamp(ratio, 1-ε, 1+ε) * advantages
policy_loss = -min(surr1, surr2).mean()

value_loss = MSE(values, returns)
entropy_loss = -entropy.mean()

loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

# Why clipping matters: Once the policy moves far enough in the good direction, the gradient goes to zero. This prevents destructive updates—the "proximal" part of PPO.

δ_t = r_t + γ V(s_{t+1}) (1 − done) − V(s_t)
A_t = δ_t + γλ (1 − done) A_{t+1}
returns_t = A_t + V(s_t)
```

λ interpolates between one-step TD (low variance, high bias) and Monte Carlo returns (high variance, low bias). λ = 0.95 is a well-calibrated default.

## Setup

```
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
mkdir -p models tb_logs
```

## Training

### Sanity check

Before training, verify the environment:

```
python sanity_check.py
```

Expected output:

```
obs shape: (122,)
action space: Discrete(6)
Random episode return: -28.40
Info: {'incidents_resolved': 3, 'threats_missed': 7, 'budget_remaining': 12.5, 'queue_length': 8}
```

#### Train

```
python train_ppo_scratch.py
```

Expected output:

```
step    2048 | episodes    41 | mean return (last 20)   -12.34 | pi_loss   0.021 | v_loss   45.213 | entropy 1.582
step    4096 | episodes    83 | mean return (last 20)    -3.11 | pi_loss   0.014 | v_loss   38.902 | entropy 1.541
...
step  200000 | episodes  4102 | mean return (last 20)    34.81 | pi_loss   0.003 | v_loss    8.112 | entropy 1.201
Saved model to models/threat_ppo_scratch.pt
```

### Monitor
```bash

tensorboard --logdir tb_logs
```

Open http://localhost:6006. Watch:


| Metric | Expected Behavior |
| :--- | :--- |
| train/mean_return | Rises, then plateaus |
| train/policy_loss | Stays small |
| train/value_loss | Falls steadily |
| train/entropy | Decays slowly |

Hyperparameters

| Parameter | Default | Notes |
| :--- | :--- | :--- |
total_timesteps | 200,000 | Raise to 500k if still improving
horizon | 2048 | Raise to 4096 for more stable GAE
gamma | 0.99 | Lower to 0.95 for shorter episodes
lam | 0.95 | 0.90–0.97 is standard
lr | 3e-4 | Lower to 1e-4 if unstable
clip_eps | 0.2 | 0.1–0.3 is standard
n_epochs | 10 | 4–10; higher risks off-policy drift
batch_size | 64 | 32–256
vf_coef | 0.5 | Value loss scale
ent_coef | 0.01 | Raise to 0.02–0.05 if entropy collapses
max_grad_norm | 0.5 | Gradient clipping


The three most impactful knobs: lr, ent_coef, and horizon.

## Evaluation
Compare against baselines

```bash
python evaluate_all.py
```

Expected output:
```
Random | return  -28.40 ± 12.11 | resolved  3.21 | missed  6.87
Heuristic | return   18.75 ±  8.42 | resolved  8.10 | missed  1.90
PPO | return   34.81 ±  6.03 | resolved  9.12 | missed  1.03
PPO improvement over heuristic: 85.7%
```

What to look for

| Comparison | What It Proves |
| :--- | :--- |
| PPO > Random | The agent learned something |
| PPO > Heuristic | The agent learned something better than hand-written rules |

Beating the heuristic is the meaningful bar.


Consistency and generalization

    Low standard deviation → reliable policy

    High standard deviation → erratic, possibly overfit

    Evaluate on unseen seeds → test generalization:

```python
run_policy(trained, episodes=100, seed=999)
```

If performance drops sharply on a new seed, the policy has overfit to the training distribution.


## Regression Testing

PPO training is not perfectly deterministic even with fixed seeds (GPU operations vary slightly). So regression tests assert that performance stays above a threshold, not exact equality.

```bash
pytest tests/ -v
```


### CI workflow (.github/workflows/test.yml):
```yaml
- name: Run RL regression tests
  run: pytest tests/ -v
```

Thresholds are set slightly below expected performance so normal variance doesn't cause false failures—but a real regression fails the build.

## Results
Policy	Mean Return	Std Dev	Resolved/Episode	Missed/Episode
Random	−28.40	12.11	3.21	6.87
Heuristic	18.75	8.42	8.10	1.90
PPO (trained)	34.81	6.03	9.12	1.03

PPO outperforms the heuristic baseline by 85.7% on mean return, while reducing missed threats per episode from 1.90 to 1.03.

The trained policy also has lower variance than either baseline—it's not just better on average, it's more consistent.


## Design Tradeoffs

| Decision | Alternative | Why I Chose This |
| --- | --- | --- |
| Deterministic reward | Stochastic reward | Cleaner learning signal; uncertainty lives in the state, not the objective |
| Discrete actions | Continuous control | Triage is categorical; PPO with Categorical is simpler and more sample-efficient |
| Budget of 100 | Larger/smaller budget | Calibrated so a heuristic strategy uses ~60%; binding but not fatal |
| From-scratch PPO | Stable-Baselines3 | Every component is visible and explainable in interviews |
| Shared-trunk network | Separate actor/critic | Parameter-efficient; the representation serves both objectives |
| λ = 0.95 | λ = 1.0 (MC) or λ = 0 (TD) | Balances bias and variance |
| Threshold-based regression tests | Exact-equality tests | GPU non-determinism makes exact equality unreliable |


## What I Learned

On RL engineering:

  -  Reward design is empirical, not theoretical. The budget and reward magnitudes required tuning against observed behavior.

  -  Delayed credit assignment is the hard part. An INVESTIGATE action gets credit for a CONTAIN that happens two steps later—GAE is what propagates that credit backward.

  -  PPO's stability comes from clipping. Without it, large advantages × large ratios produce destructive updates.

  -  On-policy is a spectrum. PPO reuses each batch for multiple epochs, which makes the data slightly off-policy; the ratio and clip are the price of that reuse.

On production ML:

  -  A trained model is an artifact that needs regression tests, just like code.

  -  TensorBoard observability (reward distributions, entropy, policy drift) is what separates a prototype from a production system.

  -  Baselines are mandatory. A policy that beats nothing proves nothing.

  -  Domain metrics (missed threats) matter as much as the RL objective (return)—they can diverge.

## Extensions

Ideas for extending the project:

| Extension | What It Demonstrates |
| --- | --- |
| Partial observability — hide some alerts from the observation | POMDP handling; closer to real SOC conditions |
| Vectorized environments — run 8 envs in parallel | Distributed rollout; leverages platform background |
| Continuous actions — Gaussian policy | Generalizes the algorithm to robotics-style tasks |
| Reward hacking detection — log action distributions | Critical judgment about RL failure modes |
| FastAPI serving — expose the trained policy as an endpoint | End-to-end "model → production" |
| Hyperparameter sweep — grid search over lr, ent_coef, horizon | Systematic tuning; produces a table for the write-up |
| Multi-agent — multiple analysts with shared budget | Coordination and credit assignment |

## References

    Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms

    Schulman, J., et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation

    Gymnasium documentation

    The 32 Implementation Details of PPO — the reference for the gain values and other implementation choices

## License

MIT License

Copyright (c) 2026 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.