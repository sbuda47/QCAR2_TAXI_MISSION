# qcar2_taxi_2026/controllers/steering_controller.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from qcar2_taxi_2026.utils.math_utils import clamp, rate_limit


XY = Tuple[float, float]


def _wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class SteeringParams:
    max_steer_rad: float = 0.40
    steer_smooth_alpha: float = 0.55
    steer_rate_limit: float = 0.06

    # Goal bias strength
    k_goal: float = 0.55           # converts heading error -> steering bias
    goal_bias_max: float = 0.18    # limit goal bias so lane stays dominant

    # If lane confidence is very low, reduce lane steer contribution slightly
    lane_conf_min: float = 0.08
    lane_conf_soften: float = 0.6  # 0..1 (lower -> more softened lane steer when conf low)


class SteeringController:
    """
    Lane-first + goal bias controller:

      steer = steer_lane + steer_goal_bias

    - steer_lane comes from LanePerception.compute(...)
    - steer_goal_bias = clamp(k_goal * heading_error, ±goal_bias_max)

    Then apply:
      - clamp to max_steer_rad
      - low-pass smoothing + rate limit
    """

    def __init__(self, params: SteeringParams):
        self.p = params
        self.prev_steer = 0.0

    def reset(self):
        self.prev_steer = 0.0

    def compute(
        self,
        pose_xy: XY,
        pose_yaw: float,
        target_xy: Optional[XY],
        lane_out: Dict,
    ) -> float:
        """
        Inputs:
          pose_xy: (x, y) in map frame
          pose_yaw: yaw in radians
          target_xy: current taxi target (x,y) or None
          lane_out: result dict from LanePerception.compute(...)

        Returns:
          final steering angle (rad)
        """
        steer_lane = float(lane_out.get("steer_lane", 0.0))
        road_conf = float(lane_out.get("road_conf", 0.0))
        yellow_conf = float(lane_out.get("yellow_conf", 0.0))

        # Lane confidence proxy
        lane_conf = max(0.0, min(1.0, 0.65 * road_conf + 0.35 * yellow_conf))

        # Optional: soften lane steer if lane is unreliable (prevents wild swings)
        if lane_conf < self.p.lane_conf_min:
            steer_lane *= float(self.p.lane_conf_soften)

        # Goal bias
        steer_goal_bias = 0.0
        if target_xy is not None:
            dx = float(target_xy[0] - pose_xy[0])
            dy = float(target_xy[1] - pose_xy[1])
            if abs(dx) + abs(dy) > 1e-6:
                desired_heading = math.atan2(dy, dx)
                e_yaw = _wrap_pi(desired_heading - pose_yaw)
                steer_goal_bias = self.p.k_goal * e_yaw
                steer_goal_bias = clamp(steer_goal_bias, -self.p.goal_bias_max, self.p.goal_bias_max)

        steer = steer_lane + steer_goal_bias

        # Clamp to steering limits
        steer = clamp(steer, -self.p.max_steer_rad, self.p.max_steer_rad)

        # Smooth + rate limit
        alpha = self.p.steer_smooth_alpha
        steer = (1.0 - alpha) * steer + alpha * self.prev_steer
        steer = rate_limit(self.prev_steer, steer, self.p.steer_rate_limit)
        self.prev_steer = steer

        return float(steer)