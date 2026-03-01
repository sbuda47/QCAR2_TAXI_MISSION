#!/usr/bin/env python3
# qcar2_taxi_2026/planning/trip_planner.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


XY = Tuple[float, float]


def dist(a: XY, b: XY) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lerp(a: XY, b: XY, t: float) -> XY:
    t = max(0.0, min(1.0, t))
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


@dataclass(frozen=True)
class Waypoint:
    id: str
    xy: XY


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    cost: float


class TripPlanner:
    """
    Graph planner (A* over fixed waypoint graph).

    Uses:
      - measured corners C1..C5
      - hub on edge C1<->C5
      - roundabout center (RND)
      - four-way intersection center (XING)

    NOTE: These are planning waypoints, not lane centerlines.
    The lane follower + traffic controller will keep the vehicle in-lane while
    the high-level route selects which "area" to head toward next.
    """

    def __init__(
        self,
        hub_xy: XY = (-1.118, -0.939),
        pickup_xy: XY = (0.125, 4.395),
        dropoff_xy: XY = (-0.905, 0.800),
        snap_radius_m: float = 1.35,
    ):
        self.hub_xy = hub_xy
        self.pickup_xy = pickup_xy
        self.dropoff_xy = dropoff_xy
        self.snap_radius_m = float(snap_radius_m)

        # -----------------------------
        # Measured corners
        # -----------------------------
        C1 = (-2.002, -0.219)
        C2 = (-2.012, 4.561)
        C3 = (2.286, 4.445)
        C4 = (2.296, -1.081)
        C5 = (-0.975, -1.256)

        # Extra measured key points
        RND = (1.561, 3.762)   # roundabout center
        XING = (0.121, 0.940)  # 4-way traffic light center

        # -----------------------------
        # Waypoints
        # -----------------------------
        w: Dict[str, Waypoint] = {}

        # Mission key points
        w["HUB"] = Waypoint("HUB", hub_xy)
        w["PICK"] = Waypoint("PICK", pickup_xy)
        w["DROP"] = Waypoint("DROP", dropoff_xy)

        # Map bounds
        w["C1"] = Waypoint("C1", C1)
        w["C2"] = Waypoint("C2", C2)
        w["C3"] = Waypoint("C3", C3)
        w["C4"] = Waypoint("C4", C4)
        w["C5"] = Waypoint("C5", C5)

        # Perimeter midpoints (helps bending)
        w["C1_C2_M"] = Waypoint("C1_C2_M", lerp(C1, C2, 0.50))
        w["C2_C3_M"] = Waypoint("C2_C3_M", lerp(C2, C3, 0.55))
        w["C3_C4_M"] = Waypoint("C3_C4_M", lerp(C3, C4, 0.55))
        w["C4_C5_M"] = Waypoint("C4_C5_M", lerp(C4, C5, 0.50))
        w["C5_C1_M"] = Waypoint("C5_C1_M", lerp(C5, C1, 0.50))

        # Key junctions
        w["RND"] = Waypoint("RND", RND)
        w["XING"] = Waypoint("XING", XING)

        # Approach points (small offsets so route can "touch" junction areas smoothly)
        # (These are guesses; refine later if needed.)
        w["X_N"] = Waypoint("X_N", (XING[0], XING[1] + 0.85))
        w["X_S"] = Waypoint("X_S", (XING[0], XING[1] - 0.85))
        w["X_E"] = Waypoint("X_E", (XING[0] + 0.85, XING[1]))
        w["X_W"] = Waypoint("X_W", (XING[0] - 0.85, XING[1]))

        w["R_S"] = Waypoint("R_S", (RND[0], RND[1] - 0.85))
        w["R_E"] = Waypoint("R_E", (RND[0] + 0.85, RND[1]))
        w["R_W"] = Waypoint("R_W", (RND[0] - 0.85, RND[1]))

        self.waypoints = w

        # -----------------------------
        # Edges (undirected)
        # -----------------------------
        self.edges: List[Edge] = []

        # Perimeter loop (with midpoints)
        self._add_edge("C1", "C1_C2_M")
        self._add_edge("C1_C2_M", "C2")

        self._add_edge("C2", "C2_C3_M")
        self._add_edge("C2_C3_M", "C3")

        self._add_edge("C3", "C3_C4_M")
        self._add_edge("C3_C4_M", "C4")

        self._add_edge("C4", "C4_C5_M")
        self._add_edge("C4_C5_M", "C5")

        self._add_edge("C5", "C5_C1_M")
        self._add_edge("C5_C1_M", "C1")

        # HUB on edge C1<->C5 
        self._add_edge("HUB", "C5_C1_M")
        self._add_edge("HUB", "C5")
        self._add_edge("HUB", "C1")

        # Pickup/dropoff anchors
        self._add_edge("PICK", "C2_C3_M")
        self._add_edge("PICK", "C3")
        self._add_edge("DROP", "X_W")      # drop is near center-ish
        self._add_edge("DROP", "C5")       # also connect to bottom-left segment

        # ---- Four-way (traffic lights) structure ----
        self._add_edge("XING", "X_N")
        self._add_edge("XING", "X_S")
        self._add_edge("XING", "X_E")
        self._add_edge("XING", "X_W")

        # Connect XING to perimeter/bottom/center areas
        self._add_edge("X_S", "C4_C5_M")
        self._add_edge("X_W", "C5")
        self._add_edge("X_E", "C4")       # right-bottom access
        self._add_edge("X_N", "C2_C3_M")  # up toward pickup area

        # ---- Roundabout structure ----
        self._add_edge("RND", "R_S")
        self._add_edge("RND", "R_E")
        self._add_edge("RND", "R_W")

        # Connect roundabout to top perimeter
        self._add_edge("R_W", "C2")       # left/top access
        self._add_edge("R_E", "C3")       # right/top access
        self._add_edge("R_S", "C2_C3_M")  # down toward middle/top corridor

        # Connect roundabout to pickup area
        self._add_edge("R_S", "PICK")

        # Also connect XING and RND (this is the main "through-map" corridor)
        self._add_edge("X_N", "R_S")

        # Build adjacency
        self.adj: Dict[str, List[Tuple[str, float]]] = {k: [] for k in self.waypoints.keys()}
        for e in self.edges:
            self.adj[e.a].append((e.b, e.cost))
            self.adj[e.b].append((e.a, e.cost))

    # -----------------------------
    # Public API
    # -----------------------------
    def plan_hub_to_pickup(self) -> List[XY]:
        return self.plan(self.hub_xy, self.pickup_xy)

    def plan_pickup_to_dropoff(self) -> List[XY]:
        return self.plan(self.pickup_xy, self.dropoff_xy)

    def plan_dropoff_to_hub(self) -> List[XY]:
        return self.plan(self.dropoff_xy, self.hub_xy)

    def plan(self, start_xy: XY, goal_xy: XY) -> List[XY]:
        start_id = self._snap_to_waypoint(start_xy)
        goal_id = self._snap_to_waypoint(goal_xy)

        if start_id is None or goal_id is None:
            return [start_xy, goal_xy]

        ids = self._astar(start_id, goal_id)
        if not ids:
            return [start_xy, goal_xy]

        route: List[XY] = [start_xy]
        for wid in ids:
            route.append(self.waypoints[wid].xy)
        route.append(goal_xy)

        return self._dedupe_close(route, eps=0.05)

    # -----------------------------
    # Internals
    # -----------------------------
    def _add_edge(self, a: str, b: str):
        if a not in self.waypoints or b not in self.waypoints:
            raise KeyError(f"Edge references unknown waypoint: {a} <-> {b}")
        self.edges.append(Edge(a, b, dist(self.waypoints[a].xy, self.waypoints[b].xy)))

    def _snap_to_waypoint(self, xy: XY) -> Optional[str]:
        best_id = None
        best_d = 1e18
        for wid, wp in self.waypoints.items():
            d = dist(xy, wp.xy)
            if d < best_d:
                best_d = d
                best_id = wid
        if best_id is None:
            return None
        if best_d > self.snap_radius_m:
            return None
        return best_id

    def _astar(self, start_id: str, goal_id: str) -> List[str]:
        open_set: List[Tuple[float, str]] = [(0.0, start_id)]
        came_from: Dict[str, str] = {}
        g: Dict[str, float] = {start_id: 0.0}
        visited = set()

        def h(nid: str) -> float:
            return dist(self.waypoints[nid].xy, self.waypoints[goal_id].xy)

        while open_set:
            open_set.sort(key=lambda x: x[0])
            _, cur = open_set.pop(0)

            if cur == goal_id:
                return self._reconstruct(came_from, cur)

            if cur in visited:
                continue
            visited.add(cur)

            for nb, cost in self.adj.get(cur, []):
                tentative = g[cur] + cost
                if tentative < g.get(nb, 1e18):
                    came_from[nb] = cur
                    g[nb] = tentative
                    open_set.append((tentative + h(nb), nb))

        return []

    @staticmethod
    def _reconstruct(came_from: Dict[str, str], cur: str) -> List[str]:
        path = [cur]
        while cur in came_from:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path

    @staticmethod
    def _dedupe_close(points: List[XY], eps: float = 0.05) -> List[XY]:
        if not points:
            return points
        out = [points[0]]
        for p in points[1:]:
            if dist(p, out[-1]) > eps:
                out.append(p)
        return out