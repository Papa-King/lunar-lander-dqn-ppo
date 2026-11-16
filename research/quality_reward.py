"""Optional training-only terminal bonus for centered, gentle safe landings."""

from __future__ import annotations

import gymnasium as gym

from metrics import PAD_MARGIN_WORLD


class QualityRewardWrapper(gym.Wrapper):
    """Modify training reward only; evaluation always uses original LunarLander-v3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.first_contact = None

    def reset(self, **kwargs):
        self.first_contact = None
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.first_contact is None and (observation[6] > 0.5 or observation[7] > 0.5):
            self.first_contact = tuple(float(observation[i]) for i in (2, 3, 4, 5))
        # The underlying environment sets exactly +100 for a safe rest.
        if terminated and float(reward) == 100.0:
            base = self.env.unwrapped
            x = float(base.lander.position.x)
            mid = (base.helipad_x1 + base.helipad_x2) / 2
            allowed_half_width = (base.helipad_x2 - base.helipad_x1) / 2 - PAD_MARGIN_WORLD
            placement = max(-1.0, 1.0 - abs(x - mid) / allowed_half_width)
            # A continuous placement signal: +60 at the center, 0 at the
            # margin boundary, negative outside that boundary.
            placement_bonus = 60.0 * placement
            gentle_bonus = 0.0
            if self.first_contact is not None:
                vx, vy, angle, omega = self.first_contact
                normalized = (abs(vx) / 0.1 + abs(vy) / 0.1 +
                              abs(angle) / 0.1 + abs(omega) / 0.25) / 4
                gentle_bonus = 30.0 * max(0.0, 1.0 - normalized)
            reward = float(reward) + placement_bonus + gentle_bonus
            info = {**info, "quality_placement_bonus": placement_bonus,
                    "quality_gentle_bonus": gentle_bonus}
        return observation, reward, terminated, truncated, info


class QualityRewardWrapperV2(QualityRewardWrapper):
    """Add a small near-ground signal for excess descent and rotation speed."""

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # y and velocities are normalized LunarLander observations. Penalize
        # excess speed only close to the landing surface so high-altitude
        # descent remains available. This is training-only shaping.
        proximity = max(0.0, 1.0 - max(0.0, float(observation[1])) / 0.5)
        downward_excess = max(0.0, -float(observation[3]) - 0.08) / 0.1
        rotation_excess = max(0.0, abs(float(observation[5])) - 0.2) / 0.2
        approach_penalty = 0.25 * proximity * (downward_excess + rotation_excess)
        reward = float(reward) - approach_penalty
        if "quality_gentle_bonus" in info:
            # Boost the terminal contact signal from a maximum of 30 to 100.
            reward += (70.0 / 30.0) * info["quality_gentle_bonus"]
        return observation, reward, terminated, truncated, info


class SlowDescentRewardWrapper(QualityRewardWrapperV2):
    """Train toward a slower approach without changing evaluation physics."""

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        height = max(0.0, float(observation[1]))
        # Allow faster descent higher up, then taper the target near the pad.
        target_downward_speed = 0.06 + 0.08 * min(1.0, height)
        proximity = max(0.0, 1.0 - height / 1.5)
        excess = max(0.0, -float(observation[3]) - target_downward_speed)
        slow_penalty = 0.8 * proximity * excess / 0.1
        info = {**info, "slow_approach_penalty": slow_penalty}
        return observation, float(reward) - slow_penalty, terminated, truncated, info
