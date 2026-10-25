# robot_arm_movement

This ROS 2 Python package controls the simulated or physical robot arm through separate arm and gripper action controllers.

## Responsibilities

The node provides:

- arm trajectory control through `/arm_controller/follow_joint_trajectory`;
- gripper control through `/gripper_controller/gripper_cmd`;
- home-pose initialization;
- tempo-scaled trajectory duration;
- camera-coordinate translation;
- randomized tap/grab behavior for interactive commands;
- independent mock execution without live controllers;
- runtime discovery of live controller joint configuration.

## Pose representation

Each command uses one five-value pose:

```text
[arm_joint_1, arm_joint_2, arm_joint_3, arm_joint_4, gripper_position]
```

The first four values are sent to the arm controller. The fifth value is sent to the gripper controller. The arm and gripper are separate ROS actions and must not receive the combined five-value list.

## Public interfaces

The node subscribes to:

| Interface | Type | Meaning |
|---|---|---|
| `/comput_game/target_action` | `std_msgs/msg/String` | Receives `interact`, `cup`, `box`, `cloth`, `reset`, or `home`. |
| `/comput_game/camera_coordinates` | `geometry_msgs/msg/Point` | Receives camera coordinates for a translated arm target. |

The node uses:

| Action | Purpose |
|---|---|
| `/arm_controller/follow_joint_trajectory` | Four-joint arm movement. |
| `/gripper_controller/gripper_cmd` | Single-joint gripper movement. |

## Command behavior

`interact`, `cup`, and `box` select randomly between tap and grab poses. `cloth`, `reset`, and `home` resolve to the home pose. Camera input uses the current fixed translation `[x, y - 0.5, z + 0.2, 0.3]`; its gripper value is selected by the tap/grab choice.

The node waits passively for incoming player or camera messages. Explicit player-state synchronization, command queueing, and command cancellation are not currently implemented.

## Dependencies

### Runtime package dependencies

The package declares these ROS 2 dependencies in `package.xml`:

- `rclpy`: node, parameter, and action-client APIs;
- `rcl_interfaces`: live controller parameter discovery;
- `geometry_msgs`: camera-coordinate messages;
- `trajectory_msgs`: arm trajectory points;
- `control_msgs`: arm and gripper action definitions;
- `std_msgs`: game command messages;
- `action_msgs`: ROS action status types.

### Live simulation dependencies

The Docker/ROS environment additionally needs `ros2_control`, `controller_manager`, the joint trajectory and gripper controllers, Gazebo ROS integration, and the OpenMANIPULATOR-X bringup/description packages. The OpenMANIPULATOR-X Dynamixel interface packages are needed for hardware-oriented configurations. These are not required for mock mode.

### Test and build dependencies

Tests use `python3-pytest` and the existing ROS 2 `ament` test tools. Building uses `ament_python`, `setuptools`, and `colcon`. Live integration testing additionally requires the simulation dependencies listed above.

## Running

Inside the ROS 2 container:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 run robot_arm_movement arm_controller_node --ros-args -p mock_mode_enabled:=true
```

For live simulation, start the OpenMANIPULATOR-X Gazebo launch first and then run the node with `mock_mode_enabled:=false`. The live node discovers the controller names and waits for both action servers before dispatching its home pose.

## Deterministic tests

The package tests deterministic pose resolution without starting ROS controllers:

```bash
python3 -m pytest -q src/robot_arm_movement/test/test_arm_controller_behavior.py
```

These tests cover home/reset mapping, deterministic tap and grab mapping, camera translation, and the four-arm-plus-one-gripper pose split. Random behavior is controlled by the test through deterministic choice injection.

## Integration tests

The reusable integration test is separate from the node implementation and exercises the complete public ROS interface:

```bash
python3 -m pytest -q -m integration src/robot_arm_movement/test/test_integration.py
```

By default it starts the node in mock mode and tests `interact`, `cup`, `box`, `cloth`, `reset`, `home`, and camera coordinates. It verifies that every command produces arm and gripper results, that results report success, and that no joint-mismatch error appears. It also verifies that `reset` and `home` retain distinct command labels while currently dispatching the same home targets.

Select the mode directly through pytest:

```bash
python3 -m pytest -q -m integration src/robot_arm_movement/test/test_integration.py --robot-arm-mode=mock
python3 -m pytest -q -m integration src/robot_arm_movement/test/test_integration.py --robot-arm-mode=live
```

The default startup and per-command timeout is 30 seconds. This allows for ROS graph discovery, executor scheduling, log flushing, and Gazebo controller response while still detecting a stuck test. Override it when the host is under load or startup is slow:

```bash
python3 -m pytest -q -m integration src/robot_arm_movement/test/test_integration.py --robot-arm-mode=live --robot-arm-timeout=60
```

Run the same test against a running live simulation with:

```bash
python3 -m pytest -q -m integration src/robot_arm_movement/test/test_integration.py --robot-arm-mode=live
```

For live mode, start Gazebo and activate both controllers before running pytest. The test requires the `ros2` CLI, the installed `robot_arm_movement` package, `rclpy`, active arm and gripper action servers, and the OpenMANIPULATOR-X simulation stack. Mock mode requires only the ROS 2 runtime and installed package. No environment variable is required.

The node's live controller-discovery timeout defaults to 15 seconds. Override it when starting the node if the simulation or hardware stack needs longer:

```bash
ros2 run robot_arm_movement arm_controller_node --ros-args \
  -p mock_mode_enabled:=false \
  -p controller_discovery_timeout_sec:=30.0
```

Start the live OpenMANIPULATOR-X simulation with the exact command used by this project:

```bash
xvfb-run -a ros2 launch open_manipulator_x_bringup gazebo.launch.py
```

Then verify the controllers in another terminal:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
ros2 control list_controllers
ros2 param get /arm_controller joints
ros2 param get /gripper_controller joint
```

The integration test is intentionally sequential: it waits for each command to appear in the node log before publishing the next command. This prevents asynchronous command overlap and makes command/result attribution meaningful. The interactive commands are randomized, so the test checks valid completion and controller separation rather than one fixed tap/grab outcome. The test is not a substitute for hardware safety validation.

## Negative and fault-injection tests

Run the deterministic failure-path tests with:

```bash
python3 -m pytest -q -m fault_injection src/robot_arm_movement/test/test_failure_paths.py
```

These tests exercise controller goal rejection, unsuccessful arm results, unsuccessful gripper results, and failed initialization. They verify that failures are logged and that the node does not falsely report initialization readiness. They use fake results and do not require Gazebo or hardware.

Live-mode discovery and action failure branches are covered with deterministic controller doubles:

```bash
python3 -m pytest -q -m fault_injection src/robot_arm_movement/test/test_live_failures.py
```

These cases cover unavailable action servers, unavailable parameter services, discovery timeout, invalid discovered joints, rejected goals, and unsuccessful arm/gripper results. They are deliberately not run against a real controller because real hardware and Gazebo do not provide deterministic failure injection. The positive live test remains necessary for real action-server and controller compatibility.

The positive integration tests verify the real ROS topic path. The fault-injection tests verify callback behavior deterministically. A future full ROS fault-injection suite may add fake action servers and parameter services to test discovery timeout, missing controller parameters, action rejection over the ROS graph, and partial initialization. A live Gazebo run is not a reliable way to generate those failures on demand.

## Known limitations

- Home initialization establishes a known home pose but does not measure or calibrate the full workspace.
- Camera translation uses fixed offsets and does not yet perform frame calibration, scaling, or workspace-bound validation.
- The current node passively waits for player messages rather than maintaining an explicit player-action state machine.

## Official references

- [ROS 2 Humble documentation](https://docs.ros.org/en/humble/)
- [ROS 2 Humble installation](https://docs.ros.org/en/humble/Installation.html)
- [ros2_control Humble documentation](https://control.ros.org/humble/)
- [ros2_control getting started](https://control.ros.org/humble/doc/getting_started/getting_started.html)
- [Gazebo ROS 2 integration](https://classic.gazebosim.org/tutorials?tut=ros2_overview)
- [OpenMANIPULATOR-X ROS 2 simulation guide](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/ros_simulation/)
- [OpenMANIPULATOR-X quick-start guide](https://emanual.robotis.com/docs/en/platform/openmanipulator_x/quick_start_guide/)
- [pytest documentation](https://docs.pytest.org/en/stable/)

The exact launch command above is project-specific but follows the official OpenMANIPULATOR-X simulation guide. This project currently uses Gazebo Classic packages in its ROS 2 Humble container; newer Gazebo distributions use different integration packages and launch conventions.
- Interactive command selection is intentionally random and therefore requires invariant-based integration assertions.
