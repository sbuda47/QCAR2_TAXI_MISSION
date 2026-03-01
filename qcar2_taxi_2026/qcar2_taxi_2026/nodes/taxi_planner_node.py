#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from qcar2_taxi_2026.planning.trip_planner import TripPlanner
from qcar2_taxi_2026.utils.ros_utils import (
    get_tf_pose_2d,
    make_goal_pose,
    stamp_now,
)
from qcar2_taxi_2026.utils.math_utils import dist2


XY = Tuple[float, float]


class TaxiPlannerNode(Node):
    """
    High-level mission planner:
      - Computes waypoint route using TripPlanner (A*)
      - Publishes current route target as PoseStamped (/taxi_target)
      - Publishes mission state as String (/taxi_state)

    It does NOT command motor commands.
    """

    def __init__(self):
        super().__init__("taxi_planner")

        # ---- Mission params ----
        self.declare_parameter("hub_xy", [-1.118, -0.939])
        self.declare_parameter("pickup_xy", [0.125, 4.395])
        self.declare_parameter("dropoff_xy", [-0.905, 0.800])

        self.declare_parameter("arrive_radius_m", 0.35)
        self.declare_parameter("hold_stop_sec", 1.5)

        # Publish topics
        self.declare_parameter("target_topic", "/taxi_target")
        self.declare_parameter("state_topic", "/taxi_state")

        # Frames for TF
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        # Planner behaviour
        self.declare_parameter("snap_radius_m", 1.35)
        self.declare_parameter("republish_hz", 5.0)  # keep target alive for subscribers

        # ---- TF ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        # ---- Publishers ----
        self.target_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_topic").value), 10
        )
        self.state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10
        )

        self.create_timer(0.5, self._publish_state)  # publish every 0.5s
        
        # ---- State machine ----
        self.state = "HUB_IDLE"
        self.state_time = self.get_clock().now()

        # Route tracking
        self.route: List[XY] = []
        self.route_idx: int = 0
        self.last_published_target: Optional[PoseStamped] = None

        # Planner object (built from params)
        self._make_planner()

        # Timers
        repub_dt = 1.0 / max(float(self.get_parameter("republish_hz").value), 0.5)
        self.create_timer(repub_dt, self._republish_target)
        self.create_timer(0.1, self.loop)

        self.get_logger().info("TaxiPlannerNode started. Publishing /taxi_target and /taxi_state")
        self._publish_state()

    # ------------------------
    # Planner setup
    # ------------------------
    def _make_planner(self):
        hub = self.get_parameter("hub_xy").value
        pick = self.get_parameter("pickup_xy").value
        drop = self.get_parameter("dropoff_xy").value

        self.planner = TripPlanner(
            hub_xy=(float(hub[0]), float(hub[1])),
            pickup_xy=(float(pick[0]), float(pick[1])),
            dropoff_xy=(float(drop[0]), float(drop[1])),
            snap_radius_m=float(self.get_parameter("snap_radius_m").value),
        )

    # ------------------------
    # Helpers
    # ------------------------
    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

    def _enter(self, new_state: str):
        self.state = new_state
        self.state_time = self.get_clock().now()
        self.get_logger().info(f"STATE -> {new_state}")
        self._publish_state()

    def _get_xy(self) -> Optional[XY]:
        pose = get_tf_pose_2d(
            self.tf_buffer, target_frame=self.map_frame, source_frame=self.base_frame
        )
        if pose is None:
            return None
        return (pose.x, pose.y)

    def _set_route(self, route: List[XY]):
        self.route = route
        self.route_idx = 0

    def _current_target_xy(self) -> Optional[XY]:
        if not self.route or self.route_idx >= len(self.route):
            return None
        return self.route[self.route_idx]

    def _advance_if_reached(self, pos: XY, r: float) -> bool:
        """
        Advance to next waypoint if within arrive radius.
        Returns True if we advanced.
        """
        tgt = self._current_target_xy()
        if tgt is None:
            return False

        if dist2(pos, tgt) <= r:
            self.route_idx += 1
            return True
        return False

    def _publish_target_xy(self, xy: XY):
        g = make_goal_pose(xy[0], xy[1], frame_id=self.map_frame, yaw=0.0)
        stamp_now(self, g)
        self.target_pub.publish(g)
        self.last_published_target = g

    def _republish_target(self):
        tgt = self._current_target_xy()
        if tgt is None:
            return
        self._publish_target_xy(tgt)

    # ------------------------
    # Main loop
    # ------------------------
    def loop(self):
        pos = self._get_xy()
        if pos is None:
            # no TF yet
            return

        arrive_r = float(self.get_parameter("arrive_radius_m").value)
        hold_sec = float(self.get_parameter("hold_stop_sec").value)

        hub = tuple(self.get_parameter("hub_xy").value)
        pickup = tuple(self.get_parameter("pickup_xy").value)
        dropoff = tuple(self.get_parameter("dropoff_xy").value)

        hub_xy = (float(hub[0]), float(hub[1]))
        pick_xy = (float(pickup[0]), float(pickup[1]))
        drop_xy = (float(dropoff[0]), float(dropoff[1]))

        # ---- State machine ----
        if self.state == "HUB_IDLE":
            # Immediately plan to pickup
            route = self.planner.plan(hub_xy, pick_xy)
            self._set_route(route)
            self._enter("GO_PICK")
            self._republish_target()

        elif self.state == "GO_PICK":
            # advance along route
            advanced = self._advance_if_reached(pos, arrive_r)
            if advanced:
                self._republish_target()

            # if final goal reached -> stop state
            if dist2(pos, pick_xy) <= arrive_r:
                self._enter("PICK_STOP")

        elif self.state == "PICK_STOP":
            age = (self.get_clock().now() - self.state_time).nanoseconds * 1e-9
            if age >= hold_sec:
                route = self.planner.plan(pick_xy, drop_xy)
                self._set_route(route)
                self._enter("GO_DROP")
                self._republish_target()

        elif self.state == "GO_DROP":
            advanced = self._advance_if_reached(pos, arrive_r)
            if advanced:
                self._republish_target()

            if dist2(pos, drop_xy) <= arrive_r:
                self._enter("DROP_STOP")

        elif self.state == "DROP_STOP":
            age = (self.get_clock().now() - self.state_time).nanoseconds * 1e-9
            if age >= hold_sec:
                route = self.planner.plan(drop_xy, hub_xy)
                self._set_route(route)
                self._enter("RETURN_HUB")
                self._republish_target()

        elif self.state == "RETURN_HUB":
            advanced = self._advance_if_reached(pos, arrive_r)
            if advanced:
                self._republish_target()

            if dist2(pos, hub_xy) <= arrive_r:
                self._enter("HUB_IDLE")


def main():
    rclpy.init()
    node = TaxiPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()