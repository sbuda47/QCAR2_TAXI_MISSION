#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from qcar2_taxi_2026.perception.traffic_yolo import TrafficYolo
from qcar2_taxi_2026.perception.sign_rules import StopLatch


def _has_label(dets, keyword: str, conf_min: float) -> bool:
    k = keyword.lower()
    for d in dets:
        if d.conf >= conf_min and k in d.label.lower():
            return True
    return False


def _best_of_labels(dets, allowed_labels, conf_min: float):
    best = None
    for d in dets:
        if d.conf < conf_min:
            continue
        if d.label.lower() in allowed_labels:
            if best is None or d.conf > best.conf:
                best = d
    return best


class TrafficDetectorNode(Node):
    """
    Subscribes: /camera/color_image
    Publishes:
      - /traffic_state (std_msgs/String): NONE / STOP / RED / YELLOW / GREEN
      - /traffic_debug_image (sensor_msgs/Image): republished raw camera msg

    Uses two YOLO models:
      - stop_model_path: COCO (stop sign)
      - light_model_path: traffic light color (red/green/yellow/off)

    Robust to occasional empty frames.
    """

    def __init__(self):
        super().__init__("traffic_detector")

        # Topics
        self.declare_parameter("image_topic", "/camera/color_image")
        self.declare_parameter("state_topic", "/traffic_state")
        self.declare_parameter("debug_image_topic", "/traffic_debug_image")
        self.declare_parameter("publish_debug_image", True)

        # Stop sign model (COCO)
        self.declare_parameter("stop_model_path", "/workspaces/isaac_ros-dev/yolov8n.pt")
        self.declare_parameter("stop_conf", 0.30)

        # Traffic light color model
        self.declare_parameter(
            "light_model_path",
            "/workspaces/isaac_ros-dev/ros2/src/mokwepa_qcar2_taxi_2026/qcar2_taxi_2026/models/traffic_light_color.pt",
        )
        self.declare_parameter("light_conf", 0.30)

        # Common YOLO settings
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("device", "cpu")

        # Stop latch
        self.declare_parameter("stop_hold_sec", 2.0)

        self.bridge = CvBridge()
        self._last_warn_ns = 0
        self.last_state = "NONE"

        dev = str(self.get_parameter("device").value)
        iou = float(self.get_parameter("iou_thres").value)

        stop_path = str(self.get_parameter("stop_model_path").value).strip()
        light_path = str(self.get_parameter("light_model_path").value).strip()

        self.stop_yolo = TrafficYolo(
            model_path=stop_path if stop_path else None,
            conf_thres=float(self.get_parameter("stop_conf").value),
            iou_thres=iou,
            device=dev,
        )

        self.light_yolo = TrafficYolo(
            model_path=light_path if light_path else None,
            conf_thres=float(self.get_parameter("light_conf").value),
            iou_thres=iou,
            device=dev,
        )

        self.stop_latch = StopLatch(hold_sec=float(self.get_parameter("stop_hold_sec").value))

        self.state_pub = self.create_publisher(String, str(self.get_parameter("state_topic").value), 10)
        self.dbg_pub = self.create_publisher(Image, str(self.get_parameter("debug_image_topic").value), 10)

        self.sub = self.create_subscription(Image, str(self.get_parameter("image_topic").value), self.image_cb, 10)

        if self.stop_yolo.available:
            self.get_logger().info(f"Stop YOLO loaded: {stop_path}")
        else:
            self.get_logger().warn(f"Stop YOLO NOT available. Check path: {stop_path}")

        if self.light_yolo.available:
            self.get_logger().info(f"Light YOLO loaded: {light_path}")
        else:
            self.get_logger().warn(f"Light YOLO NOT available. Check path: {light_path}")

        self.get_logger().info("TrafficDetectorNode started (2-model mode).")

    def _warn_throttle(self, text: str, period_sec: float = 2.0):
        now_ns = self.get_clock().now().nanoseconds
        if (now_ns - self._last_warn_ns) > int(period_sec * 1e9):
            self._last_warn_ns = now_ns
            self.get_logger().warn(text)

    def publish_state(self, state: str):
        if state != self.last_state:
            self.last_state = state
            msg = String()
            msg.data = state
            self.state_pub.publish(msg)

    def _safe_imgmsg_to_bgr(self, msg: Image):
        frame = None
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            frame = None

        if frame is None:
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            except Exception:
                frame = None

        if frame is None or not isinstance(frame, np.ndarray):
            return None

        # normalize dtype
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # ensure 3-channel
        if frame.ndim == 2:
            frame = np.repeat(frame[:, :, None], 3, axis=2)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        frame = np.ascontiguousarray(frame)

        # CRITICAL: guard empty frames (prevents YOLO division-by-zero)
        if frame.ndim != 3:
            return None
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if h <= 2 or w <= 2:
            return None

        return frame

    def image_cb(self, msg: Image):
        # republish raw camera for rqt viewing
        if bool(self.get_parameter("publish_debug_image").value):
            self.dbg_pub.publish(msg)

        frame = self._safe_imgmsg_to_bgr(msg)
        if frame is None:
            self._warn_throttle("TrafficDetector: got an empty/invalid frame; skipping YOLO.")
            return

        # YOLO inference (robust: never crash the node)
        stop_dets = []
        light_dets = []

        try:
            if self.stop_yolo.available:
                stop_dets = self.stop_yolo.infer(frame)
        except Exception as e:
            self._warn_throttle(f"Stop YOLO infer failed: {type(e).__name__}")

        try:
            if self.light_yolo.available:
                light_dets = self.light_yolo.infer(frame)
        except Exception as e:
            self._warn_throttle(f"Light YOLO infer failed: {type(e).__name__}")

        stop_seen = _has_label(stop_dets, "stop sign", conf_min=float(self.get_parameter("stop_conf").value))
        best_light = _best_of_labels(
            light_dets,
            {"red", "yellow", "green"},
            conf_min=float(self.get_parameter("light_conf").value),
        )

        # STOP latch enforcement
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        stop_now = self.stop_latch.update(now_sec, stop_seen=stop_seen)

        # Priority: STOP > RED > YELLOW > GREEN > NONE
        if stop_now:
            state = "STOP"
        elif best_light is not None:
            lab = best_light.label.lower()
            if lab == "red":
                state = "RED"
            elif lab == "yellow":
                state = "YELLOW"
            elif lab == "green":
                state = "GREEN"
            else:
                state = "NONE"
        else:
            state = "NONE"

        self.publish_state(state)


def main():
    rclpy.init()
    node = TrafficDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()