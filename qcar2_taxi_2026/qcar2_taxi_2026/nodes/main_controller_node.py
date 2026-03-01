#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge

from qcar2_interfaces.msg import MotorCommands

from qcar2_taxi_2026.perception.lane_perception import LanePerception
from qcar2_taxi_2026.controllers.steering_controller import SteeringController, SteeringParams
from qcar2_taxi_2026.controllers.speed_controller import SpeedController, SpeedParams
from qcar2_taxi_2026.utils.ros_utils import get_tf_pose_2d
from qcar2_taxi_2026.utils.math_utils import clamp


class MainController(Node):
    def __init__(self):
        super().__init__("main_controller")

        # ---------- Core params ----------
        self.declare_parameter("topics.image", "/camera/color_image")
        self.declare_parameter("topics.taxi_target", "/taxi_target")
        self.declare_parameter("topics.traffic_state", "/traffic_state")
        self.declare_parameter("topics.cmd", "/qcar2_motor_speed_cmd")

        self.declare_parameter("frames.map", "map")
        self.declare_parameter("frames.base", "base_link")

        self.declare_parameter("cmd_rate_hz", 25.0)

        self.declare_parameter("debug", False)
        self.declare_parameter("debug_print_hz", 2.0)

        # ---------- Lane params ----------
        for k, v in {
            "lane.road_roi_y0": 330,
            "lane.road_roi_y1": 480,
            "lane.road_v_max": 85,
            "lane.road_min_ratio": 0.03,
            "lane.road_steer_gain": 1.4,
            "lane.yellow_roi_y0": 180,
            "lane.yellow_roi_y1": 480,
            "lane.yellow_target_x_ratio": 0.30,
            "lane.yellow_steer_gain": 2.0,
            "lane.yellow_min_ratio": 0.001,
            "lane.yellow_col_frac": 0.35,
            "lane.yellow_memory_sec": 0.7,
            "lane.yellow_h_low": 15,
            "lane.yellow_h_high": 45,
            "lane.yellow_s_low": 60,
            "lane.yellow_v_low": 60,
            "lane.w_road": 1.30,
            "lane.w_yellow": 0.85,
            "lane.yellow_guard_enable": True,
            "lane.yellow_guard_max_x_ratio": 0.40,
            "lane.yellow_guard_push": 0.08,
            "lane.yellow_guard_k": 1.4,
        }.items():
            self.declare_parameter(k, v)

        # ---------- Steering params ----------
        for k, v in {
            "steer.max_steer_rad": 0.40,
            "steer.steer_smooth_alpha": 0.55,
            "steer.steer_rate_limit": 0.06,
            "steer.k_goal": 0.55,
            "steer.goal_bias_max": 0.18,
            "steer.lane_conf_min": 0.08,
            "steer.lane_conf_soften": 0.60,
        }.items():
            self.declare_parameter(k, v)

        # ---------- Speed params ----------
        for k, v in {
            "speed.base_throttle": 0.55,
            "speed.min_throttle": 0.38,
            "speed.max_throttle": 0.80,
            "speed.yield_factor": 0.55,
            "speed.turn_slow_k": 0.55,
            "speed.turn_slow_min": 0.55,
            "speed.lane_slow_k": 0.60,
            "speed.lane_slow_min": 0.50,
            "speed.slow_radius_m": 1.20,
            "speed.stop_radius_m": 0.35,
            "speed.approach_min_throttle": 0.10,
        }.items():
            self.declare_parameter(k, v)

        # ---------- ROS infra ----------
        self.bridge = CvBridge()
        self.last_frame = None

        self.traffic_state = "NONE"
        self.target_xy = None
        self.last_print_sec = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_frame = str(self.get_parameter("frames.map").value)
        self.base_frame = str(self.get_parameter("frames.base").value)

        # Only publisher of motor commands
        self.cmd_pub = self.create_publisher(MotorCommands, str(self.get_parameter("topics.cmd").value), 10)

        # Subscribers
        self.create_subscription(Image, str(self.get_parameter("topics.image").value), self.image_cb, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("topics.taxi_target").value), self.target_cb, 10)
        self.create_subscription(String, str(self.get_parameter("topics.traffic_state").value), self.traffic_cb, 10)

        # Build lane perception + controllers
        self.lane = LanePerception(self._lane_params_dict())

        self.steer_ctrl = SteeringController(
            SteeringParams(
                max_steer_rad=float(self.get_parameter("steer.max_steer_rad").value),
                steer_smooth_alpha=float(self.get_parameter("steer.steer_smooth_alpha").value),
                steer_rate_limit=float(self.get_parameter("steer.steer_rate_limit").value),
                k_goal=float(self.get_parameter("steer.k_goal").value),
                goal_bias_max=float(self.get_parameter("steer.goal_bias_max").value),
                lane_conf_min=float(self.get_parameter("steer.lane_conf_min").value),
                lane_conf_soften=float(self.get_parameter("steer.lane_conf_soften").value),
            )
        )

        self.speed_ctrl = SpeedController(
            SpeedParams(
                base_throttle=float(self.get_parameter("speed.base_throttle").value),
                min_throttle=float(self.get_parameter("speed.min_throttle").value),
                max_throttle=float(self.get_parameter("speed.max_throttle").value),
                yield_factor=float(self.get_parameter("speed.yield_factor").value),
                turn_slow_k=float(self.get_parameter("speed.turn_slow_k").value),
                turn_slow_min=float(self.get_parameter("speed.turn_slow_min").value),
                lane_slow_k=float(self.get_parameter("speed.lane_slow_k").value),
                lane_slow_min=float(self.get_parameter("speed.lane_slow_min").value),
                slow_radius_m=float(self.get_parameter("speed.slow_radius_m").value),
                stop_radius_m=float(self.get_parameter("speed.stop_radius_m").value),
                approach_min_throttle=float(self.get_parameter("speed.approach_min_throttle").value),
            )
        )

        hz = float(self.get_parameter("cmd_rate_hz").value)
        self.create_timer(1.0 / max(hz, 1.0), self.loop)

        self.get_logger().info("MainController started (namespaced params).")

    # ---------- callbacks ----------
    def image_cb(self, msg: Image):
        try:
            self.last_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.last_frame = None

    def target_cb(self, msg: PoseStamped):
        self.target_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def traffic_cb(self, msg: String):
        self.traffic_state = str(msg.data).strip().upper() if msg.data else "NONE"

    # ---------- param dict for LanePerception ----------
    def _lane_params_dict(self):
        keys = [
            "lane.road_roi_y0", "lane.road_roi_y1", "lane.road_v_max", "lane.road_min_ratio", "lane.road_steer_gain",
            "lane.yellow_roi_y0", "lane.yellow_roi_y1", "lane.yellow_target_x_ratio", "lane.yellow_steer_gain",
            "lane.yellow_min_ratio", "lane.yellow_col_frac", "lane.yellow_memory_sec",
            "lane.yellow_h_low", "lane.yellow_h_high", "lane.yellow_s_low", "lane.yellow_v_low",
            "lane.w_road", "lane.w_yellow",
            "lane.yellow_guard_enable", "lane.yellow_guard_max_x_ratio", "lane.yellow_guard_push", "lane.yellow_guard_k",
        ]
        # convert lane.* -> expected keys without prefix for LanePerception
        out = {}
        for k in keys:
            val = self.get_parameter(k).value
            out[k.replace("lane.", "")] = val
        return out

    def _publish_cmd(self, steer: float, throttle: float):
        msg = MotorCommands()
        msg.motor_names = ["steering_angle", "motor_throttle"]
        msg.values = [float(steer), float(throttle)]
        self.cmd_pub.publish(msg)

    def loop(self):
        if self.last_frame is None:
            self._publish_cmd(0.0, 0.0)
            return

        pose = get_tf_pose_2d(self.tf_buffer, target_frame=self.map_frame, source_frame=self.base_frame)
        if pose is None:
            self._publish_cmd(0.0, 0.0)
            return

        pose_xy = (pose.x, pose.y)
        pose_yaw = pose.yaw
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        lane_out = self.lane.compute(self.last_frame, now_sec)

        steer = self.steer_ctrl.compute(
            pose_xy=pose_xy,
            pose_yaw=pose_yaw,
            target_xy=self.target_xy,
            lane_out=lane_out,
        )

        throttle = self.speed_ctrl.compute(
            traffic_state=self.traffic_state,
            steer=steer,
            lane_out=lane_out,
            pose_xy=pose_xy,
            target_xy=self.target_xy,
        )

        steer = clamp(steer, -float(self.get_parameter("steer.max_steer_rad").value), float(self.get_parameter("steer.max_steer_rad").value))
        throttle = clamp(throttle, 0.0, float(self.get_parameter("speed.max_throttle").value))

        self._publish_cmd(steer, throttle)

        if bool(self.get_parameter("debug").value):
            hz = float(self.get_parameter("debug_print_hz").value)
            if now_sec - self.last_print_sec >= (1.0 / max(hz, 0.5)):
                self.last_print_sec = now_sec
                self.get_logger().info(
                    f"traffic={self.traffic_state} tgt={self.target_xy} "
                    f"road_conf={lane_out['road_conf']:.2f} yellow_conf={lane_out['yellow_conf']:.2f} "
                    f"steer={steer:.3f} thr={throttle:.3f}"
                )


def main():
    rclpy.init()
    node = MainController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()