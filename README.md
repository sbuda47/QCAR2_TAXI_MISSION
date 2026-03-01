**QCar2 Taxi Mission 2026 **

This repository contains a ROS 2 (Humble) control stack for the Quanser Virtual QCar2 taxi scenario (2026). The system is designed with a control-engineering mindset: clear separation of perception, planning, and control, deterministic authority over actuators, and repeatable start/stop procedures.

The single most important design rule in this project is:
-Only ONE node is allowed to publish /qcar2_motor_speed_cmd (MotorCommands).
-Any additional publisher (e.g., Nav2 converters) will cause actuator contention, erratic behavior, and unsafe trajectories.

**System Overview**

Control Objectives
-Follow the right-hand lane, keeping the yellow centerline on the left.
-Respect traffic control elements:
-Stop signs (COCO “stop sign” class)
-Traffic lights with state classification (red/yellow/green/off)

Execute taxi routing:
-Pickup at [0.125, 4.395]
-Dropoff at [-0.905, 0.800]
-Return to hub near [0.0, 0.0]

Provide an extensible framework for:
-Mission state logic (pickup/dropoff stop durations)
-LED signaling (magenta/green/blue/orange states)
-Road-rule expansions (yield handling, priority logic)

**High-Level Architecture**

Runtime nodes (launched together in Terminal 3)
When you run our project launch file, it starts all of the project nodes:
-Traffic Detector (traffic_detector)
-Runs dual YOLO inference:
-Taxi Planner (taxi_planner)
  Simple finite-state mission planner that publishes the current target waypoint.
-Main Controller (main_controller) — the only actuator authority
  Subscribes to camera, traffic state, taxi target, and uses TF for pose.
  
Executes:
-Lane perception computation (pure logic module)
-Steering controller (lane-first + goal bias)
-Speed controller (traffic-rule + curvature + confidence + approach slowdown)
Publishes:
  /qcar2_motor_speed_cmd (MotorCommands) ONLY publisher allowed
  
Why this structure works (control rationale)
No actuator contention: a single actuator node eliminates conflicts.
Separable tuning: lane/steer/speed parameters are configured centrally via YAML.
Fail-safe behavior: if inputs are missing (no TF, no image), the controller commands a stop.

**Required External Components (Platform Setup)**

This project runs across two environments:
-Quanser Virtual QCar2 Docker container (spawns QLabs map + assets)
-Isaac ROS dev container (runs ROS 2 stack and our nodes)
This README focuses on the project stack. For full environment setup, follow Quanser’s official guides:
  Virtual ROS Software Setup:
  https://github.com/quanser/student-competition-resources-ros/blob/main/Virtual_ROS_Resources/Virtual_ROS_Software_Setup.md
  Virtual ROS Development Guide:
  https://github.com/quanser/student-competition-resources-ros/blob/main/Virtual_ROS_Resources/Virtual_ROS_Development_Guide.md

**How to run**

**Terminal 1 — Quanser container (spawn map WITH signs)**

On the Ubuntu host:
  sudo docker rm -f virtual-qcar2 2>/dev/null || true
  sudo docker run --rm -it --network host --name virtual-qcar2 quanser/virtual-qcar2 bash
Inside the container:
  python3 Base_Scenarios_Python/Setup_Competition_Map.py
Verification: In QLabs you should see the track and road assets (stop signs, yield, traffic lights).

**Terminal 2 — Isaac ROS bringup (camera/scan/tf/map)**

Inside the container:
  export ROS_DOMAIN_ID=0
  source /opt/ros/humble/setup.bash
  cd /workspaces/isaac_ros-dev
  colcon build --symlink-install
  source /workspaces/isaac_ros-dev/install/setup.bash
  
Launch the QCar2 virtual SLAM + Nav bringup:
  ros2 launch qcar2_nodes qcar2_slam_and_nav_bringup_virtual_launch.py
  
Verification (core topics must exist):
  ros2 topic list | grep -E "^/camera/color_image$|^/scan$|^/tf$|^/tf_static$|^/map$|^/qcar2_motor_speed_cmd$|^/qcar2_led_cmd$"
  
Expected:
  /camera/color_image, /scan
  /tf, /tf_static
  /map
  /qcar2_motor_speed_cmd
  /qcar2_led_cmd

**Terminal 3 — Project Stack Launch (runs all project nodes)**

This is the only terminal you use to run the project nodes.
Inside the Isaac ROS container (new terminal attached to container or in the same one after bringup is running):
   source /workspaces/isaac_ros-dev/install/setup.bash
   ros2 launch qcar2_taxi_2026 qcar2_taxi_stack.launch.py \
   params_file:=/workspaces/isaac_ros-dev/ros2/src/mokwepa_qcar2_taxi_2026/qcar2_taxi_2026/config/main_controller.yaml
   
This launch starts:
  traffic_detector_1
  taxi_planner_1
  main_controller_1

**Critical Safety / Control Integrity Check (Actuator Authority)**

Before driving, confirm that only the main controller is commanding MotorCommands:
  ros2 topic info /qcar2_motor_speed_cmd -v
  
Desired state:
  Publisher count: 1
  Publisher node: main_controller_1
  
If you see nav2_qcar2_converter publishing
This is actuator contention. The converter exists to translate Nav2 /cmd_vel into QCar2 MotorCommands. Since our controller publishes MotorCommands directly, the converter must not publish.
Quick mitigation (in the Isaac ROS container):
  pkill -f nav2_qcar2_converter || true
Then re-check:
  ros2 topic info /qcar2_motor_speed_cmd -v
Note: if the converter respawns, it is being launched by the bringup. In that case, the bringup configuration must be adjusted. The --show-args list for the bringup launch does not expose a “disable converter” argument, so the practical approach is process termination or a parameter override file applied to bringup (advanced).

**What the YAML Controls **

All tunable parameters for perception + steering + speed are centralized in:
config/main_controller.yaml
We use namespaced parameters, for example:
  lane.* → perception geometry & thresholds
  steer.* → steering gains, smoothing, limits
  speed.* → throttle policy, approach slowdown, safety limits
  topics.*, frames.* → integration wiring
This keeps tuning systematic and repeatable.

**Operational Telemetry **

Traffic state output
  ros2 topic echo /traffic_state
Taxi mission state and target
  ros2 topic echo /taxi_state
  ros2 topic echo /taxi_target --once
Motor commands being applied
  ros2 topic echo /qcar2_motor_speed_cmd
Pose validity (TF)
  ros2 run tf2_ros tf2_echo map base_link

**Repository Structure**

qcar2_taxi_2026/
 qcar2_taxi_2026/
   controllers/
     speed_controller.py
     steering_controller.py
   perception/
     lane_perception.py
     traffic_yolo.py
     sign_rules.py
   nodes/
     traffic_detector_node.py
     taxi_planner_node.py
     main_controller_node.py
   utils/
     math_utils.py
     ros_utils.py
 config/
   main_controller.yaml
 launch/
   qcar2_taxi_stack.launch.py
 setup.py
 package.xml
 README.md
