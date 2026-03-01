# qcar2_taxi_2026/utils/ros_utils.py
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from qcar2_interfaces.msg import MotorCommands
from tf2_ros import Buffer

from .math_utils import yaw_from_quat


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


def make_goal_pose(x: float, y: float, frame_id: str = "map", yaw: float = 0.0) -> PoseStamped:
    """
    Create a PoseStamped goal. we set orientation as a pure yaw quaternion.
    """
    import math
    from geometry_msgs.msg import Quaternion

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = 0.0

    # quaternion from yaw
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    msg.pose.orientation = q
    return msg


def publish_motor_cmd(node: Node, pub, steer_rad: float, throttle: float) -> None:
    """
    Publish MotorCommands to /qcar2_motor_speed_cmd.
    """
    msg = MotorCommands()
    msg.motor_names = ["steering_angle", "motor_throttle"]
    msg.values = [float(steer_rad), float(throttle)]
    pub.publish(msg)


def stop_car(node: Node, pub) -> None:
    publish_motor_cmd(node, pub, 0.0, 0.0)


def get_tf_pose_2d(
    tf_buffer: Buffer,
    target_frame: str = "map",
    source_frame: str = "base_link",
) -> Optional[Pose2D]:
    """
    Return (x,y,yaw) of source_frame expressed in target_frame.
    """
    try:
        tf = tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        return Pose2D(float(x), float(y), float(yaw))
    except Exception:
        return None


def stamp_now(node: Node, msg) -> None:
    """
    Set msg.header.stamp = node's current time (works for PoseStamped etc).
    """
    msg.header.stamp = node.get_clock().now().to_msg()


def qos_is_best_effort_hint() -> str:
    """
    print when debugging /goal_pose etc.
    """
    return "If echo misses messages, try: --qos-reliability best_effort"