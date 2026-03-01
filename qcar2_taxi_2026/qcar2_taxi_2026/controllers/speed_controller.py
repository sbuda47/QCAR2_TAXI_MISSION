# qcar2_taxi_2026/controllers/speed_controller.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

from qcar2_taxi_2026.utils.math_utils import clamp, dist2


XY = Tuple[float, float]


@dataclass
class SpeedParams:
    # base speed limits
    base_throttle: float = 0.25
    min_throttle: float = 0.18
    max_throttle: float = 0.40

    # traffic behaviour
    yield_factor: float = 0.55       # used for YELLOW (and later YIELD)
    red_stop: bool = True

    # turning slowdown
    turn_slow_k: float = 0.55        # how much steering reduces speed
    turn_slow_min: float = 0.55      # minimum factor from turning

    # lane confidence slowdown (when perception is shaky)
    lane_slow_k: float = 0.60        # how much low conf reduces speed
    lane_slow_min: float = 0.50

    # approach slowdown near target
    slow_radius_m: float = 1.20      # start slowing down within this distance
    stop_radius_m: float = 0.35      # near target, crawl to stop
    approach_min_throttle: float = 0.10


class SpeedController:
    """
    Rule-based throttle controller.

    Inputs:
      - traffic_state: STOP/RED/YELLOW/GREEN/NONE
      - steer: current commanded steering magnitude
      - lane_out: dict from LanePerception (road_conf/yellow_conf)
      - pose_xy and target_xy: for approach slowdown

    Returns:
      - throttle command
    """

    def __init__(self, params: SpeedParams):
        self.p = params

    def compute(
        self,
        traffic_state: str,
        steer: float,
        lane_out: Dict,
        pose_xy: Optional[XY],
        target_xy: Optional[XY],
    ) -> float:
        traffic_state = (traffic_state or "NONE").upper()

        # 1) Hard stops
        if traffic_state in ("STOP", "RED"):
            return 0.0

        # 2) Base throttle with traffic caution
        thr = float(self.p.base_throttle)
        if traffic_state in ("YELLOW", "YIELD"):
            thr *= float(self.p.yield_factor)

        # 3) Turn-based slowdown
        steer_mag = abs(float(steer))
        # factor = 1 - k*(|steer|/max_steer_like)
        # we don't know max steer here, so we assume steer already in ~[0..0.4]
        turn_factor = 1.0 - self.p.turn_slow_k * min(1.0, steer_mag / 0.40)
        turn_factor = max(self.p.turn_slow_min, turn_factor)

        # 4) Lane-confidence slowdown
        road_conf = float(lane_out.get("road_conf", 0.0))
        yellow_conf = float(lane_out.get("yellow_conf", 0.0))
        lane_conf = max(0.0, min(1.0, 0.65 * road_conf + 0.35 * yellow_conf))

        lane_factor = 1.0 - self.p.lane_slow_k * (1.0 - lane_conf)
        lane_factor = max(self.p.lane_slow_min, lane_factor)

        thr *= min(turn_factor, lane_factor)

        # 5) Approach slowdown near target (pickup/dropoff/hub waypoint)
        if pose_xy is not None and target_xy is not None:
            d = dist2(pose_xy, target_xy)
            # very close: crawl
            if d <= self.p.stop_radius_m:
                thr = min(thr, self.p.approach_min_throttle)
            # within slow radius: scale down smoothly
            elif d <= self.p.slow_radius_m:
                # scale 0..1 between stop_radius and slow_radius
                t = (d - self.p.stop_radius_m) / max(self.p.slow_radius_m - self.p.stop_radius_m, 1e-6)
                t = clamp(t, 0.0, 1.0)
                # interpolate between approach_min and current thr
                thr = self.p.approach_min_throttle + t * (thr - self.p.approach_min_throttle)

        # Clamp final
        thr = clamp(thr, 0.0, self.p.max_throttle)
        if thr > 0.0:
            thr = max(self.p.min_throttle, thr)

        return float(thr)