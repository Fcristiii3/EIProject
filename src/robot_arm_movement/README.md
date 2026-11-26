# robot_arm_movement

ROS 2 Humble Python package controlling the OpenMANIPULATOR-X arm and gripper for the `EIProject` camera rhythm game (`"Palm on cup, fist on empty"`).

## Responsibilities

- Live URDF discovery via `/robot_state_publisher` (`robot_description` XML), extracting `joint1`–`joint4`, `gripper_left_joint` (excluding `<mimic>` `gripper_right_joint` and fixed joints), link offsets, and physical limits (`[-0.045, -0.027]` default gripper clamp, configurable via `gripper_open_position` and `gripper_closed_position`).
- Live controller & broadcaster verification via `/spawner_joint_state_broadcaster`, `/joint_states`, `/arm_controller/follow_joint_trajectory`, and `/gripper_controller/gripper_cmd`.
- Analytical 4-DOF OpenMANIPULATOR-X inverse kinematics (`solve_open_manipulator_ik`) and forward kinematics (`forward_kinematics_xyz`) in `robot_arm_movement/arm_controller_node.py` with radial clamping, pitch fallback, and configurable camera frame calibration (`camera_offset_xyz`, `camera_scale_xyz`, `camera_preferred_pitch`).
- Multi-step rhythm-game motion sequences for `EIProject`'s camera loop:
  - **`tap`** (`OPEN_GRIPPER = -0.045`): move open to target and retract to `HOME_POSE` keeping gripper open (`done:tap` -> `ready`), leaving the cup visible in the camera FOV.
  - **`grab`** (`CLOSED_GRIPPER = -0.027`): approach open -> close on cup & remember table coordinates -> lift vertically -> return to `grab_hold_pose` keeping gripper closed (`done:grab`).
  - **`place` / `reset` while holding** (or on the next camera/interactive turn while `holding_cup` is `True`, or via optional `grab_hold_duration_sec > 0`): carry back over remembered table coordinates -> lower -> release (`OPEN_GRIPPER`) -> retract to `HOME_POSE` (`done:place` -> `ready`).
- Tempo scaling via `/comput_game/tempo` and `tempo_scale`, plus game-loop synchronization via `/comput_game/arm_status`.

## Public ROS Interfaces

### Subscribed topics

| Topic | Type | Purpose |
|---|---|---|
| `/comput_game/target_action` | `std_msgs/msg/String` | `interact`, `cup`, `box`, `place`, `cloth`, `reset`, `home`. |
| `/comput_game/camera_coordinates` | `geometry_msgs/msg/Point` | Cup offset `(x, y, z)` relative to `robotBaseID=0` from `camera_node`. |
| `/comput_game/tempo` | `std_msgs/msg/Float64` | Dynamically updates `tempo_scale` (clamped `>= 0.1`). |
| `/joint_states` | `sensor_msgs/msg/JointState` | Live joint names and feedback from `joint_state_broadcaster`. |

### Published topics

| Topic | Type | Purpose |
|---|---|---|
| `/comput_game/arm_status` | `std_msgs/msg/String` | Publishes `initializing`, `ready`, `busy`, `done:tap`, `done:grab`, `done:place`, `done:reset`, `done:home`, or `error`. |
| `/comput_game/tempo` | `std_msgs/msg/Float64` | Broadcasts initial and active `tempo_scale`. |

### Action clients & service clients

| Interface | Type | Purpose |
|---|---|---|
| `/arm_controller/follow_joint_trajectory` | `control_msgs/action/FollowJointTrajectory` | Four-joint arm trajectory execution (`joint1`–`joint4`). |
| `/gripper_controller/gripper_cmd` | `control_msgs/action/GripperCommand` | Gripper execution (`gripper_left_joint`, default `-0.045` open / `-0.027` closed). |
| `/robot_state_publisher/get_parameters` | `rcl_interfaces/srv/GetParameters` | Fetches `robot_description` URDF XML on live startup. |

## Launching (`hardware`, `fake`, `gazebo`, `attach`, `mock`)

Unified launch entry point: `launch/robot_arm_movement.launch.py` (`auto_return_home:=true` by default):

```bash
# Real hardware + EIProject camera
ros2 launch robot_arm_movement robot_arm_movement.launch.py \
  mode:=hardware port_name:=/dev/ttyUSB0 launch_camera:=true

# Headless fake ros2_control stack (uses config/controllers.yaml with use_sim:=false)
ros2 launch robot_arm_movement robot_arm_movement.launch.py mode:=fake

# Gazebo Classic simulation
xvfb-run -a ros2 launch robot_arm_movement robot_arm_movement.launch.py mode:=gazebo

# Attach only arm_controller_node to an already-running controller stack
ros2 launch robot_arm_movement robot_arm_movement.launch.py mode:=attach

# Standalone mock mode (no hardware or ros2_control required)
ros2 launch robot_arm_movement robot_arm_movement.launch.py mode:=mock
```

> **Note on `open_manipulator_x_bringup fake.launch.py`:** Upstream `base.launch.py` defaults `use_sim` to `true`, which skips `ros2_control_node` unless `use_sim:=false` is passed and causes `/spawner_joint_state_broadcaster` to hang. `robot_arm_movement.launch.py mode:=fake` launches `robot_state_publisher`, `ros2_control_node` (with SI-unit `config/controllers.yaml`), and controller spawners with `use_sim:=false` directly.

### Key parameters

- `mock_mode_enabled` (`bool`, default `false` in standalone node; `true` in `mode:=mock` launch)
- `auto_return_home` (`bool`, default `true`)
- `gripper_open_position` (`double`, default `-0.045`) and `gripper_closed_position` (`double`, default `-0.027`)
- `grab_hold_duration_sec` (`double`, default `0.0`; `> 0.0` auto-places cup after hold delay)
- `grab_hold_pose` (`double[5]`, default `[0.0, -1.0, 0.0, 1.0, CLOSED_GRIPPER]`)
- `camera_offset_xyz` (`double[3]`, default `[0.0, 0.0, 0.0]`), `camera_scale_xyz` (`double[3]`, default `[1.0, -1.0, 1.0]`), `camera_preferred_pitch` (`double`, default `0.3`)
- `camera_action_mode` (`string`, default `"random"`; set `"tap"` or `"grab"` to force deterministic behavior)
- `tempo_scale` (`double`, default `1.0`), `controller_discovery_timeout_sec` (`double`, default `15.0`)

## Building & Testing

```bash
# Build inside ROS 2 container
colcon build --symlink-install --packages-select robot_arm_movement

# Full test & linter suite (unit, discovery, launch, fault injection, mock integration, flake8, pep257)
colcon test --packages-select robot_arm_movement --event-handlers console_cohesion+
colcon test-result --test-result-base build/robot_arm_movement --verbose

# Headless fake ros2_control integration suite
colcon test --packages-select robot_arm_movement --event-handlers console_cohesion+ \
  --pytest-args -m integration ' --robot-arm-mode=fake' ' --robot-arm-timeout=45'
```
