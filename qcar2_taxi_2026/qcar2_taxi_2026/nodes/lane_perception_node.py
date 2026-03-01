#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from qcar2_taxi_2026.perception.lane_perception import LanePerception


class LanePerceptionNode(Node):
    """
    Subscribes to /camera/color_image.
    Does NOT publish motor commands or steering.
    Intended to be used as a standalone perception monitor / debug tool.

    The main controller will usually import LanePerception directly and not need this node.
    """

    def __init__(self):
        super().__init__("lane_perception")

        # Params (same defaults as your previous lane follower)
        self.declare_parameter("image_topic", "/camera/color_image")

        # Perception params
        self.declare_parameter("road_roi_y0", 330)
        self.declare_parameter("road_roi_y1", 480)
        self.declare_parameter("road_v_max", 85)
        self.declare_parameter("road_min_ratio", 0.03)
        self.declare_parameter("road_steer_gain", 1.4)

        self.declare_parameter("yellow_roi_y0", 180)
        self.declare_parameter("yellow_roi_y1", 480)
        self.declare_parameter("yellow_target_x_ratio", 0.30)
        self.declare_parameter("yellow_steer_gain", 2.0)

        self.declare_parameter("yellow_min_ratio", 0.001)
        self.declare_parameter("yellow_col_frac", 0.35)
        self.declare_parameter("yellow_memory_sec", 0.7)

        self.declare_parameter("yellow_h_low", 15)
        self.declare_parameter("yellow_h_high", 45)
        self.declare_parameter("yellow_s_low", 60)
        self.declare_parameter("yellow_v_low", 60)

        self.declare_parameter("w_road", 1.30)
        self.declare_parameter("w_yellow", 0.85)

        self.declare_parameter("yellow_guard_enable", True)
        self.declare_parameter("yellow_guard_max_x_ratio", 0.40)
        self.declare_parameter("yellow_guard_push", 0.08)
        self.declare_parameter("yellow_guard_k", 1.4)

        self.declare_parameter("debug", False)
        self.declare_parameter("debug_print_hz", 2.0)

        self.bridge = CvBridge()
        self.last_frame = None
        self.last_print_sec = 0.0

        self.lp = LanePerception(self._params_dict())

        self.create_subscription(Image, str(self.get_parameter("image_topic").value), self.image_cb, 10)

        self.get_logger().info("LanePerceptionNode started (no publishing).")

    def _params_dict(self):
        keys = [
            "road_roi_y0", "road_roi_y1", "road_v_max", "road_min_ratio", "road_steer_gain",
            "yellow_roi_y0", "yellow_roi_y1", "yellow_target_x_ratio", "yellow_steer_gain",
            "yellow_min_ratio", "yellow_col_frac", "yellow_memory_sec",
            "yellow_h_low", "yellow_h_high", "yellow_s_low", "yellow_v_low",
            "w_road", "w_yellow",
            "yellow_guard_enable", "yellow_guard_max_x_ratio", "yellow_guard_push", "yellow_guard_k",
        ]
        return {k: self.get_parameter(k).value for k in keys}

    def image_cb(self, msg: Image):
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.last_frame = None
            return

        if not bool(self.get_parameter("debug").value):
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec - self.last_print_sec < (1.0 / max(float(self.get_parameter("debug_print_hz").value), 0.5)):
            return
        self.last_print_sec = now_sec

        out = self.lp.compute(self.last_frame, now_sec)
        self.get_logger().info(
            f"road_conf={out['road_conf']:.2f} yellow_conf={out['yellow_conf']:.2f} "
            f"steer_lane={out['steer_lane']:.3f} x_yellow={out['x_yellow']}"
        )


def main():
    rclpy.init()
    node = LanePerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()