#!/usr/bin/env python3
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from qcar2_interfaces.msg import MotorCommands
from cv_bridge import CvBridge


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LaneController(Node):
    """
    Robust lane controller (Virtual QCar2 2026 map):
      - Road constraint (black road surface): keeps car off white space / walls.
      - Yellow boundary constraint: keeps yellow on LEFT (car stays right of yellow).
      - White-space avoidance (strong): detects bright white regions near edges and steers away.
      - Yellow acquisition: creep forward until yellow is detected.

    Publishes ONLY /qcar2_motor_speed_cmd.
    """

    def __init__(self):
        super().__init__("lane_controller")

        # ===== Loop rate =====
        self.declare_parameter("cmd_rate_hz", 25.0)

        # ===== Speed + steering limits =====
        self.declare_parameter("motor_throttle", 0.26)     # safer while tuning
        self.declare_parameter("min_throttle", 0.10)
        self.declare_parameter("max_steer_rad", 0.40)

        # ===== Steering smoothing =====
        self.declare_parameter("steer_smooth_alpha", 0.58)
        self.declare_parameter("steer_rate_limit", 0.06)

        # ===== Yellow detection ROI =====
        self.declare_parameter("yellow_roi_y0", 140)
        self.declare_parameter("yellow_roi_y1", 480)

        # Keep yellow around this x position (ratio of width).
        # Increase -> closer to yellow; decrease -> further right (risk wall).
        self.declare_parameter("yellow_target_x_ratio", 0.30)
        self.declare_parameter("yellow_steer_gain", 1.9)

        self.declare_parameter("yellow_min_ratio", 0.001)
        self.declare_parameter("yellow_col_frac", 0.35)
        self.declare_parameter("yellow_memory_sec", 0.8)

        self.declare_parameter("yellow_h_low", 15)
        self.declare_parameter("yellow_h_high", 45)
        self.declare_parameter("yellow_s_low", 60)
        self.declare_parameter("yellow_v_low", 60)

        # Guardrail for yellow
        self.declare_parameter("yellow_guard_enable", True)
        self.declare_parameter("yellow_guard_max_x_ratio", 0.39)
        self.declare_parameter("yellow_guard_push", 0.10)
        self.declare_parameter("yellow_guard_k", 2.0)

        # ===== Acquire mode =====
        self.declare_parameter("yellow_conf_min_drive", 0.08)
        self.declare_parameter("acquire_throttle", 0.10)
        self.declare_parameter("acquire_steer", 0.05)

        # ===== Road (black surface) detection =====
        self.declare_parameter("road_roi_y0", 320)
        self.declare_parameter("road_roi_y1", 480)
        self.declare_parameter("road_v_max", 90)
        self.declare_parameter("road_steer_gain", 1.45)

        # IMPORTANT: do NOT bias right (this is what tends to push you into the wall)
        self.declare_parameter("road_center_x_ratio", 0.50)

        # Prefer centered blobs: higher = less likely to chase dark edges/shadows
        self.declare_parameter("road_centrality_lambda", 2.8)

        # ===== White-space avoidance (strong) =====
        self.declare_parameter("white_roi_y0", 300)
        self.declare_parameter("white_roi_y1", 480)
        self.declare_parameter("white_v_min", 200)
        self.declare_parameter("white_s_max", 85)
        self.declare_parameter("white_avoid_gain", 2.2)
        self.declare_parameter("white_edge_frac", 0.35)

        # If one side is very white-dominant, force a strong steer away
        self.declare_parameter("white_hard_ratio", 0.10)     # “white is really there”
        self.declare_parameter("white_hard_steer", 0.28)     # rad (bounded by max_steer)

        # ===== Blending weights =====
        self.declare_parameter("w_road", 1.3)
        self.declare_parameter("w_yellow", 0.8)
        self.declare_parameter("w_white", 2.0)

        self.declare_parameter("road_conf_min", 0.08)
        self.declare_parameter("debug", False)

        # ===== State =====
        self.bridge = CvBridge()
        self.last_frame = None
        self.prev_steer = 0.0

        self.last_yellow_err = 0.0
        self.last_yellow_time = None
        self.last_yellow_x = None

        # ===== ROS I/O =====
        self.create_subscription(Image, "/camera/color_image", self.image_cb, 10)
        self.cmd_pub = self.create_publisher(MotorCommands, "/qcar2_motor_speed_cmd", 10)

        hz = float(self.get_parameter("cmd_rate_hz").value)
        self.create_timer(1.0 / max(hz, 1.0), self.loop)

        self.get_logger().info("LaneController started (road + yellow + strong white-avoid).")

    def image_cb(self, msg: Image):
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.last_frame = None

    def publish_cmd(self, steer: float, throttle: float):
        msg = MotorCommands()
        msg.motor_names = ["steering_angle", "motor_throttle"]
        msg.values = [float(steer), float(throttle)]
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_cmd(0.0, 0.0)

    def roi_slice(self, frame, y0, y1):
        h, w, _ = frame.shape
        y0 = int(clamp(y0, 0, h - 1))
        y1 = int(clamp(y1, y0 + 1, h))
        return frame[y0:y1, :], w

    # ---------- Yellow detection ----------
    def yellow_err_with_memory(self, roi_bgr, width):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        h_low = int(self.get_parameter("yellow_h_low").value)
        h_high = int(self.get_parameter("yellow_h_high").value)
        s_low = int(self.get_parameter("yellow_s_low").value)
        v_low = int(self.get_parameter("yellow_v_low").value)

        mask = cv2.inRange(hsv, np.array([h_low, s_low, v_low]), np.array([h_high, 255, 255]))
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        min_ratio = float(self.get_parameter("yellow_min_ratio").value)

        now = self.get_clock().now()

        if ratio >= min_ratio:
            col = np.sum(mask > 0, axis=0).astype(np.float32)
            peak = float(np.max(col)) if col.size else 0.0
            if peak >= 5.0:
                thr = peak * float(self.get_parameter("yellow_col_frac").value)
                idx = np.where(col >= thr)[0]
                if idx.size > 0:
                    x_yellow = float(np.mean(idx))
                    x_des = float(self.get_parameter("yellow_target_x_ratio").value) * width
                    err = (x_des - x_yellow) / max(width, 1)

                    self.last_yellow_err = float(err)
                    self.last_yellow_time = now
                    self.last_yellow_x = x_yellow

                    conf = clamp(ratio / (min_ratio * 8.0), 0.0, 1.0)
                    return float(err), float(conf), x_yellow

        # memory fallback
        mem_sec = float(self.get_parameter("yellow_memory_sec").value)
        if self.last_yellow_time is None:
            return 0.0, 0.0, None

        age = (now - self.last_yellow_time).nanoseconds * 1e-9
        if age <= mem_sec:
            conf = clamp(1.0 - (age / max(mem_sec, 1e-6)), 0.0, 1.0)
            return float(self.last_yellow_err * conf), float(0.5 * conf), self.last_yellow_x

        return 0.0, 0.0, None

    # ---------- Road detection (robust contour selection) ----------
    def road_steer_from_black(self, roi_bgr, width):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        v_max = int(self.get_parameter("road_v_max").value)
        mask = (v < v_max).astype(np.uint8) * 255

        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0, 0.0

        desired_cx = float(self.get_parameter("road_center_x_ratio").value) * width
        lam = float(self.get_parameter("road_centrality_lambda").value)

        best_mask = None
        best_cx = None
        best_ratio = 0.0
        best_score = -1e18

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 900:
                continue

            tmp = np.zeros_like(mask)
            cv2.drawContours(tmp, [c], -1, 255, thickness=-1)
            M = cv2.moments(tmp)
            if abs(M["m00"]) < 1e-6:
                continue

            cx = M["m10"] / M["m00"]
            ratio = float(np.count_nonzero(tmp)) / float(tmp.size)

            dx = abs(cx - desired_cx) / max(width, 1)
            score = area * (1.0 - lam * dx)

            if score > best_score:
                best_score = score
                best_mask = tmp
                best_cx = cx
                best_ratio = ratio

        if best_mask is None:
            return 0.0, 0.0

        err = (desired_cx - best_cx) / max(width, 1)
        gain = float(self.get_parameter("road_steer_gain").value)
        steer = clamp(gain * err, -0.6, 0.6)

        conf = clamp(best_ratio / 0.25, 0.0, 1.0)
        return float(steer), float(conf)

    # ---------- White-space avoidance ----------
    def white_avoid(self, roi_bgr, width):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        v_min = int(self.get_parameter("white_v_min").value)
        s_max = int(self.get_parameter("white_s_max").value)

        white = ((v >= v_min) & (s <= s_max)).astype(np.uint8) * 255
        white = cv2.medianBlur(white, 5)

        edge_frac = float(self.get_parameter("white_edge_frac").value)
        edge_w = int(max(1, edge_frac * width))

        left = white[:, :edge_w]
        right = white[:, width - edge_w:]

        left_ratio = float(np.count_nonzero(left)) / float(left.size)
        right_ratio = float(np.count_nonzero(right)) / float(right.size)

        # If right is whiter -> steer LEFT (negative)
        diff = right_ratio - left_ratio

        gain = float(self.get_parameter("white_avoid_gain").value)
        steer = clamp(-gain * diff, -0.6, 0.6)

        conf = clamp((left_ratio + right_ratio) / 0.30, 0.0, 1.0)
        return steer, conf, left_ratio, right_ratio

    def loop(self):
        if self.last_frame is None:
            self.stop()
            return

        # --- Yellow ROI ---
        yy0 = int(self.get_parameter("yellow_roi_y0").value)
        yy1 = int(self.get_parameter("yellow_roi_y1").value)
        yroi, width = self.roi_slice(self.last_frame, yy0, yy1)

        yellow_err, yellow_conf, x_yellow = self.yellow_err_with_memory(yroi, width)

        # --- Road ROI ---
        ry0 = int(self.get_parameter("road_roi_y0").value)
        ry1 = int(self.get_parameter("road_roi_y1").value)
        rroi, _ = self.roi_slice(self.last_frame, ry0, ry1)
        steer_road, road_conf = self.road_steer_from_black(rroi, width)

        # --- White ROI ---
        wy0 = int(self.get_parameter("white_roi_y0").value)
        wy1 = int(self.get_parameter("white_roi_y1").value)
        wroi, _ = self.roi_slice(self.last_frame, wy0, wy1)
        steer_white, white_conf, left_white, right_white = self.white_avoid(wroi, width)

        # --- HARD white safety override ---
        white_hard_ratio = float(self.get_parameter("white_hard_ratio").value)
        white_hard_steer = float(self.get_parameter("white_hard_steer").value)
        max_steer = float(self.get_parameter("max_steer_rad").value)

        # If the right edge is strongly white -> force a left steer
        if right_white > white_hard_ratio and (right_white - left_white) > 0.03:
            steer = -min(white_hard_steer, max_steer)
            throttle = float(self.get_parameter("min_throttle").value)
            self.publish_cmd(steer, throttle)
            return

        # --- Acquire mode ---
        conf_min = float(self.get_parameter("yellow_conf_min_drive").value)
        acquire_thr = float(self.get_parameter("acquire_throttle").value)
        acquire_steer = float(self.get_parameter("acquire_steer").value)

        if yellow_conf < conf_min:
            w_road = float(self.get_parameter("w_road").value)
            w_white = float(self.get_parameter("w_white").value)
            steer_cmd = (w_road * steer_road) + (w_white * white_conf * steer_white)
            steer_cmd = clamp(steer_cmd, -acquire_steer, acquire_steer)
            self.publish_cmd(steer_cmd, acquire_thr)
            return

        # --- Yellow steer ---
        ky = float(self.get_parameter("yellow_steer_gain").value)
        steer_y = clamp(ky * yellow_err, -0.6, 0.6)

        if bool(self.get_parameter("yellow_guard_enable").value) and (x_yellow is not None):
            x_ratio = float(x_yellow) / max(float(width), 1.0)
            max_ratio = float(self.get_parameter("yellow_guard_max_x_ratio").value)
            if x_ratio > max_ratio:
                excess = x_ratio - max_ratio
                k_guard = float(self.get_parameter("yellow_guard_k").value)
                push_max = float(self.get_parameter("yellow_guard_push").value)
                push = clamp(k_guard * excess, 0.0, push_max)
                steer_y -= push

        # --- Blend ---
        w_road = float(self.get_parameter("w_road").value)
        w_yellow = float(self.get_parameter("w_yellow").value)
        w_white = float(self.get_parameter("w_white").value)

        steer = (w_road * steer_road) + (w_yellow * yellow_conf * steer_y) + (w_white * white_conf * steer_white)
        steer = clamp(steer, -max_steer, max_steer)

        # --- Smooth + rate limit ---
        alpha = float(self.get_parameter("steer_smooth_alpha").value)
        rate_lim = float(self.get_parameter("steer_rate_limit").value)

        steer_sm = (1.0 - alpha) * steer + alpha * self.prev_steer
        d = clamp(steer_sm - self.prev_steer, -rate_lim, rate_lim)
        steer_sm = self.prev_steer + d
        self.prev_steer = steer_sm
        steer = steer_sm

        # --- Throttle shaping ---
        base_v = float(self.get_parameter("motor_throttle").value)
        min_v = float(self.get_parameter("min_throttle").value)

        turn_factor = max(0.70, 1.0 - 0.45 * (abs(steer) / max(max_steer, 1e-6)))

        road_conf_min = float(self.get_parameter("road_conf_min").value)
        road_factor = clamp(0.5 + 0.5 * clamp(road_conf / max(road_conf_min, 1e-6), 0.0, 1.0), 0.35, 1.0)

        throttle = max(min_v, base_v * min(turn_factor, road_factor))

        if bool(self.get_parameter("debug").value):
            self.get_logger().info(
                f"road_conf={road_conf:.2f} steer_road={steer_road:+.3f} | "
                f"white_conf={white_conf:.2f} Lw={left_white:.3f} Rw={right_white:.3f} steer_white={steer_white:+.3f} | "
                f"y_conf={yellow_conf:.2f} x={x_yellow} y_err={yellow_err:+.3f} steer_y={steer_y:+.3f} | "
                f"steer={steer:+.3f} thr={throttle:.3f}"
            )

        self.publish_cmd(steer, throttle)


def main():
    rclpy.init()
    node = LaneController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()