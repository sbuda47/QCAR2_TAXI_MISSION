# qcar2_taxi_2026/perception/sign_rules.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .traffic_yolo import Detection


class TrafficDirective(str, Enum):
    NONE = "NONE"
    YIELD = "YIELD"
    STOP = "STOP"
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"


@dataclass
class TrafficState:
    directive: TrafficDirective
    # confidence of the chosen directive
    conf: float
    # optional extra info (e.g. "stop_sign", "traffic_light_red")
    source_label: str


@dataclass
class StopLatch:
    """
    Enforces full stop behaviour when STOP detected:
      - once stop is triggered, hold stop for hold_sec
      - only then allow moving again
    """
    active: bool = False
    start_time_sec: float = 0.0
    hold_sec: float = 2.0

    def update(self, now_sec: float, stop_seen: bool) -> bool:
        """
        Returns True if we should command STOP now.
        """
        if stop_seen and not self.active:
            self.active = True
            self.start_time_sec = now_sec

        if self.active:
            if (now_sec - self.start_time_sec) < self.hold_sec:
                return True
            # release latch after hold time
            self.active = False
        return False


def _label_is_stop(label: str) -> bool:
    l = label.lower()
    return ("stop" in l) or (l == "stop_sign")


def _label_is_yield(label: str) -> bool:
    l = label.lower()
    return ("yield" in l) or (l == "yield_sign")


def _label_is_red(label: str) -> bool:
    l = label.lower()
    return ("red" in l and "light" in l) or (l == "traffic_light_red") or (l == "red_light")


def _label_is_green(label: str) -> bool:
    l = label.lower()
    return ("green" in l and "light" in l) or (l == "traffic_light_green") or (l == "green_light")


def _label_is_yellow(label: str) -> bool:
    l = label.lower()
    return ("yellow" in l and "light" in l) or (l == "traffic_light_yellow") or (l == "amber_light")


def choose_directive(dets: List[Detection]) -> TrafficState:
    """
    Convert raw detections into a single directive.
    Priority (highest first):
      STOP, RED, YIELD, YELLOW, GREEN, NONE

    Notes:
    - In practice you may want RED higher than STOP (depends on the competition rules).
    - You can also require bounding-box area threshold / proximity logic later.
    """
    best_stop: Optional[Detection] = None
    best_red: Optional[Detection] = None
    best_yield: Optional[Detection] = None
    best_yellow: Optional[Detection] = None
    best_green: Optional[Detection] = None

    for d in dets:
        if _label_is_stop(d.label):
            if best_stop is None or d.conf > best_stop.conf:
                best_stop = d
        elif _label_is_red(d.label):
            if best_red is None or d.conf > best_red.conf:
                best_red = d
        elif _label_is_yield(d.label):
            if best_yield is None or d.conf > best_yield.conf:
                best_yield = d
        elif _label_is_yellow(d.label):
            if best_yellow is None or d.conf > best_yellow.conf:
                best_yellow = d
        elif _label_is_green(d.label):
            if best_green is None or d.conf > best_green.conf:
                best_green = d

    if best_stop is not None:
        return TrafficState(TrafficDirective.STOP, best_stop.conf, best_stop.label)
    if best_red is not None:
        return TrafficState(TrafficDirective.RED, best_red.conf, best_red.label)
    if best_yield is not None:
        return TrafficState(TrafficDirective.YIELD, best_yield.conf, best_yield.label)
    if best_yellow is not None:
        return TrafficState(TrafficDirective.YELLOW, best_yellow.conf, best_yellow.label)
    if best_green is not None:
        return TrafficState(TrafficDirective.GREEN, best_green.conf, best_green.label)

    return TrafficState(TrafficDirective.NONE, 0.0, "none")