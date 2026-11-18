import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ThreatResponseEnv(gym.Env):
    """
    A simulated security operations center (SOC) triage environment.

    The agent receives a stream of alerts and must decide, for each step,
    how to handle the highest-priority pending alert:
      - investigate it
      - contain it immediately
      - escalate to a human
      - dismiss as false positive
      - wait for more context

    Reward balances: catching real threats, avoiding wasted effort on
    false positives, and respecting a time/compute budget.
    """

    metadata = {"render_modes": ["human"]}

    # Action indices
    INVESTIGATE = 0
    CONTAIN = 1
    ESCALATE = 2
    DISMISS = 3
    WAIT = 4
    ACTION_NAMES = {
        INVESTIGATE: "INVESTIGATE",
        CONTAIN: "CONTAIN",
        ESCALATE: "ESCALATE",
        DISMISS: "DISMISS",
        WAIT: "WAIT",
    }

    def __init__(self, max_steps: int = 50, max_queue: int = 20,
                 budget: float = 100.0, render_mode=None, seed=None):
        super().__init__()
        self.max_steps = max_steps
        self.max_queue = max_queue
        self.initial_budget = budget
        self.render_mode = render_mode

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Observation space: flattened queue + budget + step fraction
        # Each alert has 6 features:
        #   [severity, source_id, asset_criticality, age, is_known_threat_hint, revealed_threat]
        self.alert_features = 6
        obs_dim = self.max_queue * self.alert_features + 2  # + budget + step_fraction
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Internal state (initialized in reset)
        self._rng = np.random.default_rng(seed)
        self.queue = None          # list of dicts
        self.budget = None
        self.step_count = None
        self.incidents_resolved = 0
        self.threats_missed = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.queue = [self._generate_alert() for _ in range(self.max_queue)]
        self.budget = self.initial_budget
        self.step_count = 0
        self.incidents_resolved = 0
        self.threats_missed = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward = 0.0
        terminated = False
        truncated = False

        # Sort queue by severity descending so the "top" alert is highest priority
        self.queue.sort(key=lambda a: a["severity"], reverse=True)
        alert = self.queue[0] if self.queue else None

        if alert is None:
            # Nothing to do — terminate
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        is_threat = alert["is_threat"]
        cost = 0.0

        # ---- Action effects -------------------------------------------
        if action == self.INVESTIGATE and alert["investigated"]:
            reward -= 500.0   # penalize redundant investigation
            cost = 100.0

        elif action == self.INVESTIGATE:
            cost = 2.0
            # Investigation reveals the truth with 90% accuracy
            revealed_threat = is_threat if self._rng.random() < 0.9 else not is_threat
            alert["investigated"] = True
            alert["revealed_threat"] = revealed_threat
            
            # Keep alert in queue so agent can act on it next step
            # (don't pop)

        elif action == self.CONTAIN:
            cost = 5.0
            self.queue.pop(0)
            if is_threat:
                reward += 10.0
                self.incidents_resolved += 1
            else:
                reward -= 5.0  # contained a benign alert — costly

        elif action == self.ESCALATE:
            cost = 3.0
            self.queue.pop(0)
            if is_threat:
                reward += 5.0  # human caught it, but slower
                self.incidents_resolved += 1
            else:
                reward -= 1.0  # wasted human time

        elif action == self.DISMISS:
            cost = 0.5
            self.queue.pop(0)
            if is_threat:
                reward -= 10.0  # missed a real threat
                self.threats_missed += 1
            else:
                reward += 1.0  # correctly dismissed a false positive

        elif action == self.WAIT:
            cost = 1.0  # time passes
            # Waiting may reveal more context: small chance severity updates
            for a in self.queue:
                a["age"] += 1
                if self._rng.random() < 0.15:
                    a["severity"] = float(np.clip(a["severity"] + self._rng.normal(0, 0.15), 0, 1))

        # ---- Budget & time --------------------------------------------
        self.budget -= cost
        self.step_count += 1

        # Apply a soft penalty as budget approaches 0
        budget_fraction = self.budget / self.initial_budget
        if budget_fraction < 0.3:
            reward -= (0.3 - budget_fraction) * 5.0   # gradual

        if self.budget <= 0:
            terminated = True  # budget exhausted

        if self.step_count >= self.max_steps:
            truncated = True

        # Penalize remaining threats if episode ends
        if terminated or truncated:
            remaining_threats = sum(1 for a in self.queue if a["is_threat"])
            reward -= 2.0 * remaining_threats
            self.threats_missed += remaining_threats

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_alert(self) -> dict:
        severity = float(self._rng.random())
        source_id = int(self._rng.integers(0, 4))         # 0..3 = EDR/SIEM/IAM/Email
        asset_criticality = float(self._rng.random())
        # Probability an alert is a real threat correlates with severity
        p_threat = 0.1 + 0.6 * severity
        is_threat = bool(self._rng.random() < p_threat)
        return {
            "severity": severity,
            "source_id": source_id,
            "asset_criticality": asset_criticality,
            "age": 0,
            "is_threat": is_threat,
            "investigated": False,
            "revealed_threat": None,
        }

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.max_queue, self.alert_features), dtype=np.float32)
        for i, a in enumerate(self.queue[: self.max_queue]):
            obs[i] = [
                a["severity"],
                a["source_id"] / 3.0,
                a["asset_criticality"],
                min(a["age"] / 10.0, 1.0),
                1.0 if a["investigated"] else 0.0,
                1.0 if a["revealed_threat"] else 0.0,
            ]
        flat = obs.flatten()
        budget_norm = np.clip(self.budget / self.initial_budget, 0.0, 1.0)
        step_norm = self.step_count / self.max_steps
        return np.concatenate([flat, [budget_norm, step_norm]]).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "incidents_resolved": self.incidents_resolved,
            "threats_missed": self.threats_missed,
            "budget_remaining": self.budget,
            "queue_length": len(self.queue),
        }

    def render(self):
        if self.render_mode != "human":
            return
        print(f"\nStep {self.step_count} | Budget {self.budget:.1f} | "
              f"Resolved {self.incidents_resolved} | Missed {self.threats_missed}")
        for i, a in enumerate(self.queue[:5]):
            tag = "THREAT" if a["is_threat"] else "benign"
            print(f"  [{i}] sev={a['severity']:.2f} src={a['source_id']} "
                  f"crit={a['asset_criticality']:.2f} ({tag})")