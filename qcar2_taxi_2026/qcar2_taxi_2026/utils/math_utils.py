# qcar2_taxi_2026/utils/math_utils.py
import math
from typing import Tuple


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Yaw (Z axis) from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation."""
    t = clamp(t, 0.0, 1.0)
    return (1.0 - t) * a + t * b


def rate_limit(prev: float, target: float, max_delta: float) -> float:
    """Limit step change per tick."""
    d = clamp(target - prev, -max_delta, max_delta)
    return prev + d