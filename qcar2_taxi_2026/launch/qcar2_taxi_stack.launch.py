#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value="",
            description="Optional YAML params file (main_controller.yaml recommended).",
        ),

        # --- Traffic detector ---
        Node(
            package="qcar2_taxi_2026",
            executable="traffic_detector",
            name="traffic_detector_1",
            output="screen",
            parameters=[params_file] if str(params_file) else [],
        ),

        # --- Taxi planner ---
        Node(
            package="qcar2_taxi_2026",
            executable="taxi_planner",
            name="taxi_planner_1",
            output="screen",
            parameters=[params_file] if str(params_file) else [],
        ),

        # --- Main controller (ONLY motor publisher) ---
        Node(
            package="qcar2_taxi_2026",
            executable="main_controller",
            name="main_controller_1",
            output="screen",
            parameters=[params_file] if str(params_file) else [],
        ),
    ])