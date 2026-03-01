import numpy as np
import cv2

from qcar2_taxi_2026.utils.math_utils import clamp


class LanePerception:
    """
    Pure lane perception module (NO ROS).
    Given a BGR frame, computes:
      - steer_road, road_conf
      - yellow_err, yellow_conf, x_yellow
      - steer_lane (blended suggestion)
    """

    def __init__(self, params: dict):
        self.p = params

        # yellow memory
        self.last_yellow_err = 0.0
        self.last_yellow_time_sec = None
        self.last_yellow_x = None

    @staticmethod
    def roi_slice(frame, y0, y1):
        h, w, _ = frame.shape
        y0 = int(clamp(y0, 0, h - 1))
        y1 = int(clamp(y1, y0 + 1, h))
        return frame[y0:y1, :], w

    def road_steer_from_black(self, roi_bgr, width):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        v_max = int(self.p["road_v_max"])
        mask = (v < v_max).astype(np.uint8) * 255

        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0.0, 0.0

        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 800:
            return 0.0, 0.0

        road_mask = np.zeros_like(mask)
        cv2.drawContours(road_mask, [c], -1, 255, thickness=-1)

        ratio = float(np.count_nonzero(road_mask)) / float(road_mask.size)
        if ratio < float(self.p["road_min_ratio"]):
            return 0.0, 0.0

        M = cv2.moments(road_mask)
        if abs(M["m00"]) < 1e-6:
            return 0.0, 0.0

        cx = M["m10"] / M["m00"]
        desired = 0.50 * width
        err = (desired - cx) / max(width, 1)

        steer = clamp(float(self.p["road_steer_gain"]) * err, -0.6, 0.6)
        conf = clamp(ratio / 0.30, 0.0, 1.0)
        return float(steer), float(conf)

    def yellow_err_with_memory(self, roi_bgr, width, now_sec: float):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        lower = np.array([int(self.p["yellow_h_low"]), int(self.p["yellow_s_low"]), int(self.p["yellow_v_low"])])
        upper = np.array([int(self.p["yellow_h_high"]), 255, 255])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        min_ratio = float(self.p["yellow_min_ratio"])

        if ratio >= min_ratio:
            col = np.sum(mask > 0, axis=0).astype(np.float32)
            peak = float(np.max(col)) if col.size else 0.0
            if peak >= 5.0:
                thr = peak * float(self.p["yellow_col_frac"])
                idx = np.where(col >= thr)[0]
                if idx.size > 0:
                    x_yellow = float(np.mean(idx))
                    x_des = float(self.p["yellow_target_x_ratio"]) * width
                    err = (x_des - x_yellow) / max(width, 1)

                    self.last_yellow_err = float(err)
                    self.last_yellow_time_sec = now_sec
                    self.last_yellow_x = x_yellow

                    conf = clamp(ratio / (min_ratio * 8.0), 0.0, 1.0)
                    return float(err), float(conf), x_yellow

        # fallback memory
        mem = float(self.p["yellow_memory_sec"])
        if self.last_yellow_time_sec is None:
            return 0.0, 0.0, None

        age = now_sec - float(self.last_yellow_time_sec)
        if age <= mem:
            conf = clamp(1.0 - (age / max(mem, 1e-6)), 0.0, 1.0)
            return float(self.last_yellow_err * conf), float(0.5 * conf), self.last_yellow_x

        return 0.0, 0.0, None

    def compute(self, frame_bgr: np.ndarray, now_sec: float):
        # ROIs
        road_roi, w = self.roi_slice(frame_bgr, int(self.p["road_roi_y0"]), int(self.p["road_roi_y1"]))
        y_roi, _ = self.roi_slice(frame_bgr, int(self.p["yellow_roi_y0"]), int(self.p["yellow_roi_y1"]))

        steer_road, road_conf = self.road_steer_from_black(road_roi, w)
        y_err, y_conf, x_yellow = self.yellow_err_with_memory(y_roi, w, now_sec)

        steer_y = clamp(float(self.p["yellow_steer_gain"]) * y_err, -0.6, 0.6)

        # guardrail (push right if yellow too far right)
        if bool(self.p["yellow_guard_enable"]) and (x_yellow is not None):
            x_ratio = float(x_yellow) / max(float(w), 1.0)
            max_ratio = float(self.p["yellow_guard_max_x_ratio"])
            if x_ratio > max_ratio:
                excess = x_ratio - max_ratio
                push = clamp(float(self.p["yellow_guard_k"]) * excess, 0.0, float(self.p["yellow_guard_push"]))
                steer_y -= push

        w_road = float(self.p["w_road"])
        w_yellow = float(self.p["w_yellow"])
        steer_lane = (w_road * steer_road) + (w_yellow * y_conf * steer_y)

        return {
            "steer_lane": float(steer_lane),
            "steer_road": float(steer_road),
            "road_conf": float(road_conf),
            "yellow_err": float(y_err),
            "yellow_conf": float(y_conf),
            "x_yellow": None if x_yellow is None else float(x_yellow),
            "img_width": int(w),
        }