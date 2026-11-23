"""ROS 2 node controlling OpenMANIPULATOR-X arm and gripper for EIProject rhythm game."""

import functools, random, sys, rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from trajectory_msgs.msg import JointTrajectoryPoint
from robot_arm_movement.discovery import discover_live_runtime
from robot_arm_movement.kinematics import CLOSED_GRIPPER, DEFAULT_SPEC, GRAB_HOLD_POSE, GRAB_POSE, HOME_POSE, OPEN_GRIPPER, TAP_POSE, MotionStep, build_grab_sequence, build_place_sequence, build_tap_sequence, clamp_gripper, clamp_pose, solve_open_manipulator_ik


class _RejectedGoalResult:
    __slots__ = ("error_code",)

    def __init__(self):
        self.error_code = -1


class ComputRhythmArmController(Node):
    """Dual-controller OpenMANIPULATOR-X arm/gripper node with URDF discovery and 4-DOF IK."""

    def __init__(self):
        super().__init__("comput_rhythm_arm_controller")
        self._arm_action_client = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
        self._gripper_action_client = ActionClient(self, GripperCommand, "/gripper_controller/gripper_cmd")
        self.target_sub = self.create_subscription(String, "/comput_game/target_action", self.game_action_callback, 10)
        self.coord_sub = self.create_subscription(Point, "/comput_game/camera_coordinates", self.camera_coord_callback, 10)
        self.tempo_sub = self.create_subscription(Float64, "/comput_game/tempo", self.tempo_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.status_pub = self.create_publisher(String, "/comput_game/arm_status", 10)
        self.tempo_pub = self.create_publisher(Float64, "/comput_game/tempo", 10)
        self.declare_parameter("tempo_scale", 1.0)
        self.declare_parameter("mock_mode_enabled", True)
        self.declare_parameter("mock_execution_delay", 0.3)
        self.declare_parameter("controller_discovery_timeout_sec", 15.0)
        self.declare_parameter("auto_return_home", False)
        self.declare_parameter("grab_hold_duration_sec", 0.0)
        self.declare_parameter("grab_hold_pose", list(GRAB_HOLD_POSE))
        self.declare_parameter("camera_offset_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("camera_scale_xyz", [1.0, 1.0, 1.0])
        self.declare_parameter("camera_preferred_pitch", 0.3)
        self.declare_parameter("camera_action_mode", "random")
        self.spec = DEFAULT_SPEC
        self.arm_joint_names = None
        self.gripper_joint_name = None
        self.runtime_ready = False
        self.joint_states_received = False
        self.latest_joint_names = []
        self.latest_joint_positions = {}
        self.home_pose = list(HOME_POSE)
        self.tap_pose = list(TAP_POSE)
        self.grab_pose = list(GRAB_POSE)
        self.grab_hold_pose = list(GRAB_HOLD_POSE)
        self.pose_split_arm_gripper = 4
        self.available_choices = ["tap", "grab"]
        self.interactive_commands = ["interact", "cup", "box"]
        self.reset_commands = ["place", "cloth", "reset", "home"]
        self.holding_cup = False
        self._held_cup_arm_pose = None
        self._initialization_goals_pending = 2
        self._sequence_active = False
        self._pending_steps = []
        self._command_queue = []
        self._active_step_pending = 0
        self._active_step_failed = False
        self._sequence_final_status = None
        self._pending_held_arm = None
        self._grab_release_timer = None
        self._mock_timer = None
        self._discover_runtime()
        self._publish_status("initializing")
        self._publish_tempo(self._get_tempo_scale())
        self.get_logger().info("ComputRhythmArmController initialized in Dual-Mode. Executing initial calibration / home setup.")
        self.send_trajectory_goal(self.home_pose, 2.0, command_name="home", callback=self._initialization_goal_done_callback)

    def _publish_status(self, status: str):
        pub = getattr(self, "status_pub", None)
        if pub is not None: pub.publish(String(data=status))

    def _publish_tempo(self, tempo: float):
        pub = getattr(self, "tempo_pub", None)
        if pub is not None: pub.publish(Float64(data=float(tempo)))

    def _get_tempo_scale(self) -> float:
        return max(0.1, float(self.get_parameter("tempo_scale").get_parameter_value().double_value))

    def _safe_param_array(self, name: str, fallback):
        try: vals = list(self.get_parameter(name).get_parameter_value().double_array_value)
        except Exception: return list(fallback)
        return vals if vals else list(fallback)

    def _safe_param_double(self, name: str, fallback: float) -> float:
        try: return float(self.get_parameter(name).get_parameter_value().double_value)
        except Exception: return float(fallback)

    def _safe_param_bool(self, name: str, fallback: bool) -> bool:
        try: return bool(self.get_parameter(name).get_parameter_value().bool_value)
        except Exception: return bool(fallback)

    def _safe_param_str(self, name: str, fallback: str) -> str:
        try: val = self.get_parameter(name).get_parameter_value().string_value
        except Exception: return fallback
        return val if val else fallback

    def joint_state_callback(self, msg: JointState):
        """Track live joint names and positions published by joint_state_broadcaster."""
        self.latest_joint_names = list(msg.name)
        self.latest_joint_positions = dict(zip(msg.name, msg.position))
        self.joint_states_received = True

    def tempo_callback(self, msg: Float64):
        """Update tempo_scale dynamically from /comput_game/tempo."""
        new_tempo = max(0.1, float(msg.data))
        if abs(new_tempo - self._get_tempo_scale()) < 1e-6: return
        self.set_parameters([Parameter("tempo_scale", Parameter.Type.DOUBLE, new_tempo)])
        self.get_logger().info(f"Tempo scale updated via topic -> tempo_scale={new_tempo:.3f}")

    def initialization_done_callback(self, future):
        """Log readiness and notify game loop after home calibration succeeds."""
        self.get_logger().info("Arm initialization range established successfully. Ready for rhythm game commands.")
        self._publish_status("ready")
        self._drain_command_queue()

    def _initialization_goal_done_callback(self, future):
        if future is not None and hasattr(future, "error_code") and future.error_code != 0:
            self.get_logger().error(f"Initialization arm goal failed: error_code={future.error_code}.")
            self._publish_status("error")
            return
        if future is not None and hasattr(future, "reached_goal") and (not future.reached_goal or future.stalled):
            self.get_logger().error(f"Initialization gripper goal failed: reached_goal={future.reached_goal}, stalled={future.stalled}.")
            self._publish_status("error")
            return
        self._initialization_goals_pending -= 1
        if self._initialization_goals_pending == 0: self.initialization_done_callback(future)

    def _discover_runtime(self):
        if self.get_parameter("mock_mode_enabled").get_parameter_value().bool_value:
            self.get_logger().info("Mock mode enabled; skipping live controller discovery.")
            self.spec = DEFAULT_SPEC
            self.arm_joint_names = list(self.spec.arm_joint_names)
            self.gripper_joint_name = self.spec.gripper_joint_name
            self.runtime_ready = True
            return
        self.get_logger().info("Waiting for /robot_state_publisher, /joint_states, and controllers to be ready...")
        timeout_sec = self.get_parameter("controller_discovery_timeout_sec").get_parameter_value().double_value
        runtime, error_msg = discover_live_runtime(self, self._arm_action_client, self._gripper_action_client, timeout_sec=timeout_sec, expected_arm_count=self.pose_split_arm_gripper)
        if runtime is None:
            self.get_logger().error(error_msg or "Live runtime discovery failed.")
            self.runtime_ready = False
            self._publish_status("error")
            return
        self.spec = runtime.spec
        self.arm_joint_names = runtime.arm_joint_names
        self.gripper_joint_name = runtime.gripper_joint_name
        self.runtime_ready = True
        self.get_logger().info(f"Discovered live URDF runtime -> arm={self.arm_joint_names}, gripper={self.gripper_joint_name}, gripper_limits={self.spec.gripper_limits}, observed_joint_states={runtime.observed_joint_names}.")

    def _choose_action(self) -> str:
        mode = self._safe_param_str("camera_action_mode", "random").strip().lower()
        return mode if mode in self.available_choices else random.choice(self.available_choices)

    def _resolve_target_pose(self, action_type, tempo_scale, msg_coords=None):
        chosen_action = self._choose_action()
        self.get_logger().info(f'Randomized behavior choice -> action: "{chosen_action}", mode: "{action_type}", tempo_scale: {tempo_scale}')
        spec = getattr(self, "spec", DEFAULT_SPEC)
        if action_type == "camera":
            offset = self._safe_param_array("camera_offset_xyz", (0.0, 0.0, 0.0))
            scale = self._safe_param_array("camera_scale_xyz", (1.0, 1.0, 1.0))
            pitch = self._safe_param_double("camera_preferred_pitch", 0.3)
            arm = solve_open_manipulator_ik(msg_coords.x, msg_coords.y, msg_coords.z, spec=spec, preferred_pitch=pitch, camera_offset_xyz=offset, camera_scale_xyz=scale)
            grip = clamp_gripper(OPEN_GRIPPER if chosen_action == "tap" else CLOSED_GRIPPER, spec)
            return arm + [grip]
        if action_type == "interactive": return clamp_pose(self.tap_pose if chosen_action == "tap" else self.grab_pose, spec)
        return clamp_pose(self.home_pose, spec)

    def _plan_action(self, action_type: str, command_name: str, tempo_scale: float, msg_coords=None):
        auto_seq = self._safe_param_bool("auto_return_home", False)
        spec = getattr(self, "spec", DEFAULT_SPEC)
        if action_type in ("camera", "interactive"):
            pose = self._resolve_target_pose(action_type, tempo_scale, msg_coords=msg_coords)
            arm = pose[:4]
            if pose[4] >= 0.0: return build_tap_sequence(command_name, arm, home_pose=self.home_pose, spec=spec, auto_sequence_enabled=auto_seq), "done:tap", None
            hold_pose = self._safe_param_array("grab_hold_pose", self.grab_hold_pose)
            return build_grab_sequence(command_name, arm, grab_hold_pose=hold_pose, spec=spec, auto_sequence_enabled=auto_seq), "done:grab", list(arm)
        if command_name == "place" or (command_name == "reset" and self.holding_cup):
            if self.holding_cup:
                held = self._held_cup_arm_pose if self._held_cup_arm_pose is not None else list(self.grab_pose[:4])
                return build_place_sequence(command_name, held, home_pose=self.home_pose, spec=spec), "done:place", None
            return [MotionStep(command_name, clamp_pose(self.home_pose, spec), 1.0)], "done:place", None
        final_status = "done:home" if command_name == "home" else "done:reset"
        return [MotionStep(command_name, clamp_pose(self.home_pose, spec), 1.0)], final_status, None

    def _cancel_grab_release_timer(self):
        if self._grab_release_timer is not None:
            self._grab_release_timer.cancel()
            self._grab_release_timer = None

    def _auto_place_timer_callback(self):
        self._cancel_grab_release_timer()
        if not self.holding_cup: return
        steps, final_status, held_arm = self._plan_action("reset", "place", self._get_tempo_scale())
        self._dispatch_sequence(steps, final_status, held_arm)

    def _dispatch_sequence(self, steps, final_status: str, held_arm):
        self._cancel_grab_release_timer()
        if self._initialization_goals_pending > 0 or self._sequence_active:
            self._command_queue.append((steps, final_status, held_arm))
            return
        self._sequence_active = True
        self._pending_steps = list(steps)
        self._sequence_final_status = final_status
        self._pending_held_arm = held_arm
        self._publish_status("busy")
        self._start_next_step()

    def _drain_command_queue(self):
        if not self._sequence_active and self._initialization_goals_pending == 0 and self._command_queue:
            steps, final_status, held_arm = self._command_queue.pop(0)
            self._dispatch_sequence(steps, final_status, held_arm)

    def _start_next_step(self):
        if not self._pending_steps:
            self._sequence_active = False
            if self._sequence_final_status == "done:grab":
                self.holding_cup = True
                self._held_cup_arm_pose = self._pending_held_arm
                self._publish_status("done:grab")
                hold_sec = self._safe_param_double("grab_hold_duration_sec", 0.0)
                if hold_sec > 0.0: self._grab_release_timer = self.create_timer(hold_sec, self._auto_place_timer_callback)
            else:
                if self._sequence_final_status in ("done:place", "done:reset", "done:home"): self.holding_cup = False; self._held_cup_arm_pose = None
                self._publish_status(self._sequence_final_status or "ready")
                self._publish_status("ready")
            self._drain_command_queue()
            return
        step = self._pending_steps.pop(0)
        duration_sec = max(0.15, step.duration_factor * self._get_tempo_scale())
        self._active_step_pending = 2
        self._active_step_failed = False
        self.send_trajectory_goal(step.pose, duration_sec, command_name=step.name, callback=self._on_step_controller_done)

    def _on_step_controller_done(self, result):
        if result is not None and hasattr(result, "error_code") and result.error_code != 0: self._active_step_failed = True
        if result is not None and hasattr(result, "reached_goal") and (not result.reached_goal or result.stalled): self._active_step_failed = True
        self._active_step_pending -= 1
        if self._active_step_pending > 0: return
        if self._active_step_failed:
            self._pending_steps.clear()
            self._sequence_active = False
            self._publish_status("error")
            self._drain_command_queue()
            return
        self._start_next_step()

    def camera_coord_callback(self, msg: Point):
        """Translate camera ArUco coordinates via 4-DOF IK and execute tap/grab sequence."""
        self.get_logger().info(f"Camera coordinates received -> x: {msg.x:.3f}, y: {msg.y:.3f}, z: {msg.z:.3f}")
        steps, final_status, held_arm = self._plan_action("camera", "camera", self._get_tempo_scale(), msg_coords=msg)
        self._dispatch_sequence(steps, final_status, held_arm)

    def game_action_callback(self, msg: String):
        """Handle game command strings on /comput_game/target_action."""
        command = msg.data.strip().lower()
        self.get_logger().info(f'Game state command received -> command: "{command}"')
        if command not in self.interactive_commands and command not in self.reset_commands:
            self.get_logger().warn(f'Invalid game command received -> command: "{command}". Aborting execution.')
            return
        action_type = "interactive" if command in self.interactive_commands else "reset"
        steps, final_status, held_arm = self._plan_action(action_type, command, self._get_tempo_scale())
        self._dispatch_sequence(steps, final_status, held_arm)

    def _complete_mock_goal(self, callback):
        self.get_logger().info(f"Mock {self._mock_controller} goal completed: command={self._mock_command}, target={self._mock_target}.")
        if callback: callback(None)

    def _complete_mock_execution(self):
        if self._mock_timer is not None:
            self._mock_timer.cancel()
            self._mock_timer = None
        cb = self._mock_callback
        self._mock_controller = "arm"
        self._complete_mock_goal(cb)
        self._mock_controller = "gripper"
        self._mock_target = self._mock_gripper_target
        self._complete_mock_goal(cb)

    def send_trajectory_goal(self, positions, duration_sec, command_name="command", callback=None, max_effort=5.0):
        """Clamp and dispatch separate arm trajectory and gripper command goals."""
        clamped = clamp_pose(positions, getattr(self, "spec", DEFAULT_SPEC))
        arm_positions = clamped[:self.pose_split_arm_gripper]
        gripper_position = clamped[self.pose_split_arm_gripper]
        mock_enabled = self.get_parameter("mock_mode_enabled").get_parameter_value().bool_value
        if mock_enabled:
            self._mock_command = command_name
            self._mock_target = arm_positions
            self._mock_gripper_target = gripper_position
            self._mock_controller = "arm"
            self._mock_callback = callback
            self._mock_timer = self.create_timer(self.get_parameter("mock_execution_delay").get_parameter_value().double_value, self._complete_mock_execution)
            return
        if not self.runtime_ready:
            self.get_logger().error("Action server offline and mock_mode_enabled is False! Dropping goal dispatch.")
            if callback: callback(_RejectedGoalResult()); callback(_RejectedGoalResult())
            return
        self.get_logger().info(f"Command sent: name={command_name}, arm_target={arm_positions}, gripper_target={gripper_position}.")
        arm_goal = FollowJointTrajectory.Goal()
        arm_goal.trajectory.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = arm_positions
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        arm_point.time_from_start = Duration(sec=sec, nanosec=nanosec)
        arm_goal.trajectory.points.append(arm_point)
        self.get_logger().info(f"Gripper control Action server online. Dispatching live trajectory goal -> target_positions: {gripper_position}, max_effort: {max_effort:.2f}N.")
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position = float(gripper_position)
        gripper_goal.command.max_effort = max_effort
        arm_goal_future = self._arm_action_client.send_goal_async(arm_goal, feedback_callback=self.feedback_callback)
        arm_goal_future.add_done_callback(functools.partial(self.goal_response_callback, controller="arm", command_name=command_name, target=arm_positions, callback=callback))
        gripper_goal_future = self._gripper_action_client.send_goal_async(gripper_goal, feedback_callback=self.feedback_callback)
        gripper_goal_future.add_done_callback(functools.partial(self.goal_response_callback, controller="gripper", command_name=command_name, target=gripper_position, callback=callback))

    def goal_response_callback(self, future, controller, command_name, target, callback=None):
        """Handle action goal acceptance or rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Command rejected: name={command_name}, controller={controller}, target={target}.")
            if callback: callback(_RejectedGoalResult())
            return
        self.get_logger().info(f"Command accepted: name={command_name}, controller={controller}, target={target}.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(functools.partial(self.get_result_callback, controller=controller, command_name=command_name, target=target, callback=callback))

    def feedback_callback(self, feedback_msg):
        """Log action execution feedback."""
        self.get_logger().info(f"Live action feedback received -> {feedback_msg.feedback}")

    def get_result_callback(self, future, controller, command_name, target, callback=None):
        """Log final action execution result and invoke completion callback."""
        result = future.result().result
        if controller == "arm":
            success = result.error_code == 0
            self.get_logger().info(f"Command result: name={command_name}, controller=arm, target={target}, achieved_error_code={result.error_code}, success={success}.")
        else:
            success = result.reached_goal and not result.stalled
            self.get_logger().info(f"Command result: name={command_name}, controller=gripper, target={target}, achieved_position={result.position}, success={success}.")
        if callback: callback(result)


def main(args=None):
    """Run the ComputRhythmArmController ROS 2 node."""
    rclpy.init(args=args)
    node = ComputRhythmArmController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info("KeyboardInterrupt detected. Shutting down node cleanly...")
    except Exception as e: node.get_logger().error(f"An exception occurred during node spin -> exception: {e}")
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
