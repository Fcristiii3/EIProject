"""ROS 2 node, 4-DOF kinematics, and URDF discovery for OpenMANIPULATOR-X arm and gripper."""

import functools, math, random, sys, time, xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Point
from rcl_interfaces.srv import GetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from trajectory_msgs.msg import JointTrajectoryPoint

GRIPPER_MIN_RAD, GRIPPER_MAX_RAD = -0.027,-0.045
CLOSED_GRIPPER, OPEN_GRIPPER =   GRIPPER_MIN_RAD, GRIPPER_MAX_RAD
HOME_POSE = [0.0, -1.0, 0.0, 1.0, OPEN_GRIPPER]
TAP_POSE = [0.0, -1.0, 0.0, 1.0, CLOSED_GRIPPER]
GRAB_POSE = [0.0, -1.0, 0.0, 1.0, OPEN_GRIPPER]
GRAB_HOLD_POSE = [0.0, -1.0, 0.0, 1.0, CLOSED_GRIPPER]


@dataclass(frozen=True)
class RobotKinematicsSpec:
    """URDF-derived link offsets, joint names, and physical joint limits."""

    arm_joint_names: Tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4")
    gripper_joint_name: str = "gripper_left_joint"
    arm_limits: Tuple[Tuple[float, float], ...] = ((-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97))
    gripper_limits: Tuple[float, float] = (GRIPPER_MIN_RAD, GRIPPER_MAX_RAD)
    joint1_origin: Tuple[float, float, float] = (0.012, 0.0, 0.017)
    joint2_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0595)
    joint3_origin: Tuple[float, float, float] = (0.024, 0.0, 0.128)
    joint4_origin: Tuple[float, float, float] = (0.124, 0.0, 0.0)
    end_effector_origin: Tuple[float, float, float] = (0.126, 0.0, 0.0)


DEFAULT_SPEC = RobotKinematicsSpec()


@dataclass(frozen=True)
class MotionStep:
    """Single trajectory and gripper step in a multi-step arm sequence."""

    name: str
    pose: List[float]
    duration_factor: float = 1.0


@dataclass(frozen=True)
class DiscoveredRuntime:
    """Validated live robot runtime configuration discovered from ROS 2."""

    spec: RobotKinematicsSpec
    arm_joint_names: List[str]
    gripper_joint_name: str
    joint_states_received: bool
    observed_joint_names: List[str]


def clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value to inclusive [low, high] range."""
    return max(float(low), min(float(high), float(value)))


def clamp_gripper(value: float, spec: RobotKinematicsSpec = DEFAULT_SPEC) -> float:
    """Clamp gripper position strictly within the configured working range."""
    return clamp(value, spec.gripper_limits[0], spec.gripper_limits[1])


def clamp_arm_joints(joints: Sequence[float], spec: RobotKinematicsSpec = DEFAULT_SPEC) -> List[float]:
    """Clamp all four arm joints strictly within URDF revolute limits."""
    return [clamp(joints[i], spec.arm_limits[i][0], spec.arm_limits[i][1]) for i in range(4)]


def clamp_pose(pose: Sequence[float], spec: RobotKinematicsSpec = DEFAULT_SPEC) -> List[float]:
    """Clamp a 5-element [joint1..joint4, gripper] pose to physical limits."""
    return clamp_arm_joints(pose[:4], spec) + [clamp_gripper(pose[4], spec)]


def forward_kinematics_xyz(arm_joints: Sequence[float], spec: RobotKinematicsSpec = DEFAULT_SPEC) -> Tuple[float, float, float]:
    """Compute end_effector_link (x, y, z) in link1 frame for 4 arm joint angles."""
    q1, q2, q3, q4 = (float(arm_joints[i]) for i in range(4))
    x1, y1, z1 = spec.joint1_origin; x2, y2, z2 = spec.joint2_origin; x3, _, z3 = spec.joint3_origin; x4, _, z4 = spec.joint4_origin; xe, _, ze = spec.end_effector_origin
    p3, p4 = q2 + q3, q2 + q3 + q4
    r = x2 + (x3 * math.cos(q2) + z3 * math.sin(q2)) + (x4 * math.cos(p3) + z4 * math.sin(p3)) + (xe * math.cos(p4) + ze * math.sin(p4))
    z = z1 + z2 + (-x3 * math.sin(q2) + z3 * math.cos(q2)) + (-x4 * math.sin(p3) + z4 * math.cos(p3)) + (-xe * math.sin(p4) + ze * math.cos(p4))
    return (x1 + r * math.cos(q1) - y2 * math.sin(q1), y1 + r * math.sin(q1) + y2 * math.cos(q1), z)


def solve_open_manipulator_ik(x: float, y: float, z: float, spec: RobotKinematicsSpec = DEFAULT_SPEC, preferred_pitch: float = 0.3, camera_offset_xyz: Sequence[float] = (0.0, 0.0, 0.0), camera_scale_xyz: Sequence[float] = (1.0, -1.0, 1.0)) -> List[float]:
    """Solve analytical 4-DOF OpenMANIPULATOR-X IK with radial clamping and pitch fallback."""
    tx = float(x) * float(camera_scale_xyz[0]) + float(camera_offset_xyz[0])
    ty = float(y) * float(camera_scale_xyz[1]) + float(camera_offset_xyz[1])
    tz = float(z) * float(camera_scale_xyz[2]) + float(camera_offset_xyz[2])
    x1, y1, z1 = spec.joint1_origin; x2, _, z2 = spec.joint2_origin; x3, _, z3 = spec.joint3_origin; x4, _, z4 = spec.joint4_origin; xe, _, ze = spec.end_effector_origin
    dx, dy = tx - x1, ty - y1
    r_xy = math.hypot(dx, dy)
    q1_raw = math.atan2(dy, dx) if r_xy > 1e-6 else 0.0
    r_target, z_target = r_xy - x2, tz - (z1 + z2)
    l2, alpha = math.hypot(x3, z3), math.atan2(x3, z3)
    l3, beta = math.hypot(x4, z4), math.atan2(z4, x4)
    d_min, d_max = abs(l2 - l3) + 1e-4, l2 + l3 - 1e-4
    pitches = []
    for p in (preferred_pitch, 0.3, 0.5, 0.1, 0.0, 0.8, 1.0, -0.2, 1.2):
        if p not in pitches: pitches.append(p)
    best_pose, best_score = None, (float("inf"), float("inf"))
    for phi in pitches:
        rw, zw = r_target - (xe * math.cos(phi) + ze * math.sin(phi)), z_target - (-xe * math.sin(phi) + ze * math.cos(phi))
        d = math.hypot(rw, zw)
        if d < 1e-6: rw, zw, d = 0.0, d_min, d_min
        elif d < d_min or d > d_max: scale = clamp(d, d_min, d_max) / d; rw *= scale; zw *= scale; d = clamp(d, d_min, d_max)
        cos_delta = clamp((d * d - l2 * l2 - l3 * l3) / (2.0 * l2 * l3), -1.0, 1.0)
        gamma = math.atan2(rw, zw)
        for elbow_sign in (1.0, -1.0):
            delta = elbow_sign * math.acos(cos_delta)
            psi = math.atan2(l3 * math.sin(delta), l2 + l3 * math.cos(delta))
            q2, q3 = (gamma - psi) - alpha, delta - (math.pi / 2.0) + alpha + beta
            q4 = phi - q2 - q3
            raw = [q1_raw, q2, q3, q4]
            clamped = clamp_arm_joints(raw, spec)
            fx, fy, fz = forward_kinematics_xyz(clamped, spec)
            err = math.sqrt((fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2)
            if all(spec.arm_limits[i][0] - 1e-6 <= raw[i] <= spec.arm_limits[i][1] + 1e-6 for i in range(4)) and err < 1e-3: return clamped
            score = (round(err, 6), abs(phi - preferred_pitch))
            if score < best_score: best_score, best_pose = score, clamped
    return best_pose if best_pose is not None else clamp_arm_joints([q1_raw, -0.4, 0.2, 0.3], spec)


def _lift_pose(arm_pose: Sequence[float], gripper_pos: float, spec: RobotKinematicsSpec) -> List[float]:
    return clamp_arm_joints([arm_pose[0], -0.8, 0.1, 0.7], spec) + [clamp_gripper(gripper_pos, spec)]


def build_tap_sequence(cmd: str, arm_pose: Sequence[float], home_pose: Sequence[float] = HOME_POSE, spec: RobotKinematicsSpec = DEFAULT_SPEC, auto_sequence_enabled: bool = True) -> List[MotionStep]:
    """Build single-step or auto-retracting tap sequence keeping the cup on the table."""
    tap_step = MotionStep(cmd, clamp_arm_joints(arm_pose, spec) + [clamp_gripper(OPEN_GRIPPER, spec)], 1.0)
    return [tap_step] if not auto_sequence_enabled else [tap_step, MotionStep(f"{cmd}_return", clamp_pose(home_pose, spec), 1.0)]


def build_grab_sequence(cmd: str, arm_pose: Sequence[float], grab_hold_pose: Sequence[float] = GRAB_HOLD_POSE, spec: RobotKinematicsSpec = DEFAULT_SPEC, auto_sequence_enabled: bool = True) -> List[MotionStep]:
    """Build approach -> close -> vertical lift -> side hold sequence clearing the cup from FOV."""
    open_pos, closed_pos = clamp_gripper(OPEN_GRIPPER, spec), clamp_gripper(CLOSED_GRIPPER, spec)
    arm = clamp_arm_joints(arm_pose, spec)
    if not auto_sequence_enabled: return [MotionStep(cmd, arm + [closed_pos], 1.0)]
    hold = clamp_arm_joints(grab_hold_pose[:4], spec) + [closed_pos]
    return [MotionStep(f"{cmd}_approach", arm + [open_pos], 0.8), MotionStep(cmd, arm + [closed_pos], 0.6), MotionStep(f"{cmd}_lift", _lift_pose(arm, closed_pos, spec), 0.7), MotionStep(f"{cmd}_hold", hold, 1.0)]


def build_place_sequence(cmd: str, held_arm_pose: Sequence[float], home_pose: Sequence[float] = HOME_POSE, spec: RobotKinematicsSpec = DEFAULT_SPEC) -> List[MotionStep]:
    """Build carry-back -> lower -> release -> home sequence returning the cup to the table."""
    open_pos, closed_pos = clamp_gripper(OPEN_GRIPPER, spec), clamp_gripper(CLOSED_GRIPPER, spec)
    held = clamp_arm_joints(held_arm_pose, spec)
    return [MotionStep(f"{cmd}_carry", _lift_pose(held, closed_pos, spec), 1.0), MotionStep(f"{cmd}_lower", held + [closed_pos], 0.8), MotionStep(f"{cmd}_release", held + [open_pos], 0.6), MotionStep(cmd, clamp_pose(home_pose, spec), 1.0)]


def _parse_xyz(text: Optional[str], fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if not text: return fallback
    parts = [float(v) for v in text.split()]
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else fallback


def _parse_limit(joint_elem: ET.Element, fallback: Tuple[float, float]) -> Tuple[float, float]:
    limit = joint_elem.find("limit")
    if limit is None: return fallback
    low, high = float(limit.attrib.get("lower", fallback[0])), float(limit.attrib.get("upper", fallback[1]))
    return (low, high) if low <= high else fallback


def _resolve_gripper_limits(joint_elem: ET.Element) -> Tuple[float, float]:
    low, high = _parse_limit(joint_elem, (GRIPPER_MIN_RAD, GRIPPER_MAX_RAD))
    if low >= GRIPPER_MIN_RAD - 1e-6 and high <= GRIPPER_MAX_RAD + 1e-6 and low <= high: return (max(GRIPPER_MIN_RAD, low), min(GRIPPER_MAX_RAD, high))
    return (GRIPPER_MIN_RAD, GRIPPER_MAX_RAD)


def _order_kinematic_chain(joints: List[dict]) -> List[dict]:
    if not joints: return []
    child_links = {j["child"] for j in joints}
    roots = [j for j in joints if j["parent"] not in child_links]
    by_parent: Dict[str, dict] = {j["parent"]: j for j in joints}
    ordered, current, seen = [], roots[0] if roots else joints[0], set()
    while current is not None and current["name"] not in seen:
        ordered.append(current); seen.add(current["name"]); current = by_parent.get(current["child"])
    for j in joints:
        if j["name"] not in seen: ordered.append(j)
    return ordered


def parse_robot_description(urdf_xml: str, expected_arm_count: int = 4) -> RobotKinematicsSpec:
    """Parse /robot_state_publisher URDF XML into RobotKinematicsSpec without mimic or fixed joints."""
    if not urdf_xml or not urdf_xml.strip(): raise ValueError("Empty robot_description XML.")
    root = ET.fromstring(urdf_xml)
    arm_candidates, gripper_candidates, origins_by_name = [], [], {}
    for joint in root.findall("joint"):
        name, jtype, origin_elem = joint.attrib.get("name", ""), joint.attrib.get("type", ""), joint.find("origin")
        xyz = _parse_xyz(origin_elem.attrib.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        if name: origins_by_name[name] = xyz
        if jtype == "fixed" or joint.find("mimic") is not None: continue
        parent_elem, child_elem = joint.find("parent"), joint.find("child")
        parent = parent_elem.attrib.get("link", "") if parent_elem is not None else ""
        child = child_elem.attrib.get("link", "") if child_elem is not None else ""
        is_gripper = jtype == "prismatic" or "gripper" in name.lower()
        limit = _resolve_gripper_limits(joint) if is_gripper else _parse_limit(joint, (-math.pi, math.pi))
        entry = {"name": name, "type": jtype, "parent": parent, "child": child, "origin": xyz, "limit": limit}
        if is_gripper: gripper_candidates.append(entry)
        elif jtype in ("revolute", "continuous"): arm_candidates.append(entry)
    ordered_arm = _order_kinematic_chain(arm_candidates)
    if len(ordered_arm) != expected_arm_count or not gripper_candidates: raise ValueError(f"Invalid URDF joints: arm={[j['name'] for j in ordered_arm]}, gripper={[j['name'] for j in gripper_candidates]}.")
    gripper = gripper_candidates[0]
    ee_origin = next((v for k, v in origins_by_name.items() if k.endswith("end_effector_joint")), DEFAULT_SPEC.end_effector_origin)
    return RobotKinematicsSpec(arm_joint_names=tuple(j["name"] for j in ordered_arm), gripper_joint_name=gripper["name"], arm_limits=tuple(j["limit"] for j in ordered_arm), gripper_limits=gripper["limit"], joint1_origin=ordered_arm[0]["origin"], joint2_origin=ordered_arm[1]["origin"], joint3_origin=ordered_arm[2]["origin"], joint4_origin=ordered_arm[3]["origin"], end_effector_origin=ee_origin)


def _spawner_diagnostic(node) -> str:
    get_names = getattr(node, "get_node_names", None)
    if not callable(get_names): return ""
    try: names = set(get_names())
    except Exception: return ""
    has_spawner, has_cm = any("spawner_joint_state_broadcaster" in n for n in names), any("controller_manager" in n for n in names)
    if has_spawner and not has_cm: return " /spawner_joint_state_broadcaster is active while /controller_manager is missing (check Dynamixel serial port/baudrate -3001/-3002 or fake.launch.py use_sim:=false)."
    if has_spawner: return " /spawner_joint_state_broadcaster is active; verify /controller_manager loaded joint_state_broadcaster, arm_controller, and gripper_controller."
    return ""


def _wait_for_joint_states(node, timeout_sec: float) -> Tuple[bool, List[str]]:
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while not getattr(node, "joint_states_received", False) and time.monotonic() < deadline:
        try: rclpy.spin_once(node, timeout_sec=0.1)
        except Exception: time.sleep(0.05)
    return bool(getattr(node, "joint_states_received", False)), list(getattr(node, "latest_joint_names", []))


def discover_live_runtime(node, arm_action_client, gripper_action_client, timeout_sec: float = 15.0, expected_arm_count: int = 4, joint_states_topic: str = "/joint_states") -> Tuple[Optional[DiscoveredRuntime], Optional[str]]:
    """Discover URDF kinematics from /robot_state_publisher, verify /joint_states, and wait for action servers."""
    rsp_client = node.create_client(GetParameters, "/robot_state_publisher/get_parameters")
    if not rsp_client.wait_for_service(timeout_sec=timeout_sec): return None, f"/robot_state_publisher/get_parameters service is unavailable.{_spawner_diagnostic(node)}"
    rsp_future = rsp_client.call_async(GetParameters.Request(names=["robot_description"]))
    rclpy.spin_until_future_complete(node, rsp_future, timeout_sec=timeout_sec)
    if not rsp_future.done(): return None, "/robot_state_publisher robot_description discovery timed out."
    resp = rsp_future.result()
    urdf_xml = resp.values[0].string_value if resp and getattr(resp, "values", None) else ""
    try: spec = parse_robot_description(urdf_xml, expected_arm_count=expected_arm_count)
    except (ValueError, ET.ParseError) as exc: return None, f"Invalid robot_description from /robot_state_publisher: {exc}"
    arm_ready, gripper_ready = arm_action_client.wait_for_server(timeout_sec=timeout_sec), gripper_action_client.wait_for_server(timeout_sec=timeout_sec)
    if not arm_ready or not gripper_ready: return None, f"Action server availability: arm={arm_ready}, gripper={gripper_ready}.{_spawner_diagnostic(node)}"
    received, observed = _wait_for_joint_states(node, timeout_sec=min(timeout_sec, 5.0))
    if not received: return None, f"Timed out waiting for {joint_states_topic} from joint_state_broadcaster.{_spawner_diagnostic(node)}"
    missing = [name for name in list(spec.arm_joint_names) + [spec.gripper_joint_name] if name not in observed]
    if missing: return None, f"Joint mismatch between /robot_state_publisher and {joint_states_topic}: missing={missing}, observed={observed}."
    return DiscoveredRuntime(spec=spec, arm_joint_names=list(spec.arm_joint_names), gripper_joint_name=spec.gripper_joint_name, joint_states_received=True, observed_joint_names=observed), None


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
        self.declare_parameter("tempo_scale", 1.0); self.declare_parameter("mock_mode_enabled", False); self.declare_parameter("mock_execution_delay", 0.3)
        self.declare_parameter("controller_discovery_timeout_sec", 15.0); self.declare_parameter("auto_return_home", True); self.declare_parameter("grab_hold_duration_sec", 0.0)
        self.declare_parameter("grab_hold_pose", list(GRAB_HOLD_POSE)); self.declare_parameter("camera_offset_xyz", [0.0, 0.0, 0.0]); self.declare_parameter("camera_scale_xyz", [1.0, -1.0, 1.0])
        self.declare_parameter("camera_preferred_pitch", 0.3); self.declare_parameter("camera_action_mode", "random")
        self.spec, self.arm_joint_names, self.gripper_joint_name = DEFAULT_SPEC, None, None
        self.runtime_ready, self.joint_states_received, self.latest_joint_names, self.latest_joint_positions = False, False, [], {}
        self.home_pose, self.tap_pose, self.grab_pose, self.grab_hold_pose = list(HOME_POSE), list(TAP_POSE), list(GRAB_POSE), list(GRAB_HOLD_POSE)
        self.pose_split_arm_gripper, self.available_choices = 4, ["tap", "grab"]
        self.interactive_commands, self.reset_commands = ["interact", "cup", "box"], ["place", "cloth", "reset", "home"]
        self.holding_cup, self._held_cup_arm_pose, self._initialization_goals_pending = False, None, 2
        self._sequence_active, self._pending_steps, self._command_queue = False, [], []
        self._active_step_pending, self._active_step_failed, self._sequence_final_status, self._pending_held_arm = 0, False, None, None
        self._grab_release_timer, self._mock_timer = None, None
        self._discover_runtime()
        self._publish_status("initializing"); self._publish_tempo(self._get_tempo_scale())
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
        self.latest_joint_names, self.latest_joint_positions, self.joint_states_received = list(msg.name), dict(zip(msg.name, msg.position)), True

    def tempo_callback(self, msg: Float64):
        """Update tempo_scale dynamically from /comput_game/tempo."""
        new_tempo = max(0.1, float(msg.data))
        if abs(new_tempo - self._get_tempo_scale()) < 1e-6: return
        self.set_parameters([Parameter("tempo_scale", Parameter.Type.DOUBLE, new_tempo)])
        self.get_logger().info(f"Tempo scale updated via topic -> tempo_scale={new_tempo:.3f}")

    def initialization_done_callback(self, future):
        """Log readiness and notify game loop after home calibration succeeds."""
        self.get_logger().info("Arm initialization range established successfully. Ready for rhythm game commands.")
        self._publish_status("ready"); self._drain_command_queue()

    def _initialization_goal_done_callback(self, future):
        if future is not None and hasattr(future, "error_code") and future.error_code != 0:
            self.get_logger().warn(f"Initialization arm goal reported error: error_code={future.error_code}.")
        if future is not None and hasattr(future, "reached_goal") and (not future.reached_goal or future.stalled):
            self.get_logger().warn(f"Initialization gripper stalled/incomplete: reached_goal={getattr(future, 'reached_goal', None)}, stalled={getattr(future, 'stalled', None)}.")

        # Always decrement pending counter to prevent command queue lockout
        self._initialization_goals_pending -= 1
        if self._initialization_goals_pending <= 0:
            self._initialization_goals_pending = 0
            self.initialization_done_callback(future)

    def _discover_runtime(self):
        if self.get_parameter("mock_mode_enabled").get_parameter_value().bool_value:
            self.get_logger().info("Mock mode enabled; skipping live controller discovery.")
            self.spec = DEFAULT_SPEC; self.arm_joint_names = list(self.spec.arm_joint_names); self.gripper_joint_name = self.spec.gripper_joint_name; self.runtime_ready = True
            return
        self.get_logger().info("Waiting for /robot_state_publisher, /joint_states, and controllers to be ready...")
        timeout_sec = self.get_parameter("controller_discovery_timeout_sec").get_parameter_value().double_value
        runtime, error_msg = discover_live_runtime(self, self._arm_action_client, self._gripper_action_client, timeout_sec=timeout_sec, expected_arm_count=self.pose_split_arm_gripper)
        if runtime is None:
            self.get_logger().error(error_msg or "Live runtime discovery failed."); self.runtime_ready = False; self._publish_status("error"); return
        self.spec, self.arm_joint_names, self.gripper_joint_name, self.runtime_ready = runtime.spec, runtime.arm_joint_names, runtime.gripper_joint_name, True
        self.get_logger().info(f"Discovered live URDF runtime -> arm={self.arm_joint_names}, gripper={self.gripper_joint_name}, gripper_limits={self.spec.gripper_limits}, observed_joint_states={runtime.observed_joint_names}.")

    def _choose_action(self) -> str:
        mode = self._safe_param_str("camera_action_mode", "random").strip().lower()
        return mode if mode in self.available_choices else random.choice(self.available_choices)

    def _resolve_target_pose(self, action_type, tempo_scale, msg_coords=None):
        chosen_action = self._choose_action()
        self.get_logger().info(f'Randomized behavior choice -> action: "{chosen_action}", mode: "{action_type}", tempo_scale: {tempo_scale}')
        spec = getattr(self, "spec", DEFAULT_SPEC)
        if action_type == "camera":
            offset, scale, pitch = self._safe_param_array("camera_offset_xyz", (0.0, 0.0, 0.0)), self._safe_param_array("camera_scale_xyz", (1.0, -1.0, 1.0)), self._safe_param_double("camera_preferred_pitch", 0.3)
            arm = solve_open_manipulator_ik(msg_coords.x, msg_coords.y, msg_coords.z, spec=spec, preferred_pitch=pitch, camera_offset_xyz=offset, camera_scale_xyz=scale)
            return arm + [clamp_gripper(OPEN_GRIPPER if chosen_action == "tap" else CLOSED_GRIPPER, spec)]
        if action_type == "interactive": return clamp_pose(self.tap_pose if chosen_action == "tap" else self.grab_pose, spec)
        return clamp_pose(self.home_pose, spec)

    def _plan_action(self, action_type: str, command_name: str, tempo_scale: float, msg_coords=None):
        auto_seq, spec = self._safe_param_bool("auto_return_home", True), getattr(self, "spec", DEFAULT_SPEC)
        if action_type in ("camera", "interactive"):
            pose = self._resolve_target_pose(action_type, tempo_scale, msg_coords=msg_coords)
            arm, midpoint = pose[:4], 0.5 * (spec.gripper_limits[0] + spec.gripper_limits[1])
            if pose[4] >= midpoint: return build_tap_sequence(command_name, arm, home_pose=self.home_pose, spec=spec, auto_sequence_enabled=auto_seq), "done:tap", None
            hold_pose = self._safe_param_array("grab_hold_pose", self.grab_hold_pose)
            return build_grab_sequence(command_name, arm, grab_hold_pose=hold_pose, spec=spec, auto_sequence_enabled=auto_seq), "done:grab", list(arm)
        if command_name == "place" or (command_name == "reset" and self.holding_cup):
            if self.holding_cup:
                held = self._held_cup_arm_pose if self._held_cup_arm_pose is not None else list(self.grab_pose[:4])
                return build_place_sequence(command_name, held, home_pose=self.home_pose, spec=spec), "done:place", None
            return [MotionStep(command_name, clamp_pose(self.home_pose, spec), 1.0)], "done:place", None
        return [MotionStep(command_name, clamp_pose(self.home_pose, spec), 1.0)], "done:home" if command_name == "home" else "done:reset", None

    def _cancel_grab_release_timer(self):
        if self._grab_release_timer is not None: self._grab_release_timer.cancel(); self._grab_release_timer = None

    def _auto_place_timer_callback(self):
        self._cancel_grab_release_timer()
        if not self.holding_cup: return
        self._dispatch_sequence(*self._plan_action("reset", "place", self._get_tempo_scale()))

    def _dispatch_sequence(self, steps, final_status: str, held_arm):
        self._cancel_grab_release_timer()
        if self._initialization_goals_pending > 0 or self._sequence_active: self._command_queue.append((steps, final_status, held_arm)); return
        self._sequence_active, self._pending_steps, self._sequence_final_status, self._pending_held_arm = True, list(steps), final_status, held_arm
        self._publish_status("busy"); self._start_next_step()

    def _drain_command_queue(self):
        if not self._sequence_active and self._initialization_goals_pending == 0 and self._command_queue:
            self._dispatch_sequence(*self._command_queue.pop(0))

    def _start_next_step(self):
        if not self._pending_steps:
            self._sequence_active = False
            if self._sequence_final_status == "done:grab":
                self.holding_cup, self._held_cup_arm_pose = True, self._pending_held_arm
                self._publish_status("done:grab")
                hold_sec = self._safe_param_double("grab_hold_duration_sec", 0.0)
                if hold_sec > 0.0: self._grab_release_timer = self.create_timer(hold_sec, self._auto_place_timer_callback)
            else:
                if self._sequence_final_status in ("done:place", "done:reset", "done:home"): self.holding_cup, self._held_cup_arm_pose = False, None
                self._publish_status(self._sequence_final_status or "ready"); self._publish_status("ready")
            self._drain_command_queue()
            return
        step = self._pending_steps.pop(0)
        duration_sec = max(0.15, step.duration_factor * self._get_tempo_scale())
        self._active_step_pending, self._active_step_failed = 2, False
        self.send_trajectory_goal(step.pose, duration_sec, command_name=step.name, callback=self._on_step_controller_done)

    def _on_step_controller_done(self, result):
        if result is not None and hasattr(result, "error_code") and result.error_code != 0: self._active_step_failed = True
        if result is not None and hasattr(result, "reached_goal") and (not result.reached_goal or result.stalled): self._active_step_failed = True
        self._active_step_pending -= 1
        if self._active_step_pending > 0: return
        if self._active_step_failed:
            self._pending_steps.clear(); self._sequence_active = False; self._publish_status("error"); self._drain_command_queue(); return
        self._start_next_step()

    def camera_coord_callback(self, msg: Point):
        """Translate camera ArUco coordinates via 4-DOF IK and execute tap/grab sequence."""
        self.get_logger().info(f"Camera coordinates received -> x: {msg.x:.3f}, y: {msg.y:.3f}, z: {msg.z:.3f}")
        self._dispatch_sequence(*self._plan_action("camera", "camera", self._get_tempo_scale(), msg_coords=msg))

    def game_action_callback(self, msg: String):
        """Handle game command strings on /comput_game/target_action."""
        command = msg.data.strip().lower()
        self.get_logger().info(f'Game state command received -> command: "{command}"')
        if command not in self.interactive_commands and command not in self.reset_commands:
            self.get_logger().warn(f'Invalid game command received -> command: "{command}". Aborting execution.'); return
        action_type = "interactive" if command in self.interactive_commands else "reset"
        self._dispatch_sequence(*self._plan_action(action_type, command, self._get_tempo_scale()))

    def _complete_mock_execution(self):
        if self._mock_timer is not None: self._mock_timer.cancel(); self._mock_timer = None
        cb = self._mock_callback
        for ctrl, target in (("arm", self._mock_target), ("gripper", self._mock_gripper_target)):
            self.get_logger().info(f"Mock {ctrl} goal completed: command={self._mock_command}, target={target}.")
            if cb: cb(None)

    def send_trajectory_goal(self, positions, duration_sec, command_name="command", callback=None, max_effort=5.0):
        """Clamp and dispatch separate arm trajectory and gripper command goals."""
        clamped = clamp_pose(positions, getattr(self, "spec", DEFAULT_SPEC))
        arm_positions, gripper_position = clamped[:self.pose_split_arm_gripper], clamped[self.pose_split_arm_gripper]
        if self.get_parameter("mock_mode_enabled").get_parameter_value().bool_value:
            self._mock_command, self._mock_target, self._mock_gripper_target, self._mock_callback = command_name, arm_positions, gripper_position, callback
            self._mock_timer = self.create_timer(self.get_parameter("mock_execution_delay").get_parameter_value().double_value, self._complete_mock_execution)
            return
        if not self.runtime_ready:
            self.get_logger().error("Action server offline and mock_mode_enabled is False! Dropping goal dispatch.")
            if callback: callback(_RejectedGoalResult()); callback(_RejectedGoalResult())
            return
        self.get_logger().info(f"Command sent: name={command_name}, arm_target={arm_positions}, gripper_target={gripper_position}.")
        arm_goal, arm_point = FollowJointTrajectory.Goal(), JointTrajectoryPoint()
        arm_goal.trajectory.joint_names, arm_point.positions = self.arm_joint_names, arm_positions
        sec = int(duration_sec)
        arm_point.time_from_start = Duration(sec=sec, nanosec=int((duration_sec - sec) * 1e9))
        arm_goal.trajectory.points.append(arm_point)
        self.get_logger().info(f"Gripper control Action server online. Dispatching live trajectory goal -> target_positions: {gripper_position}, max_effort: {max_effort:.2f}N.")
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position, gripper_goal.command.max_effort = float(gripper_position), max_effort
        for ctrl, client, goal, target in (("arm", self._arm_action_client, arm_goal, arm_positions), ("gripper", self._gripper_action_client, gripper_goal, gripper_position)):
            client.send_goal_async(goal, feedback_callback=self.feedback_callback).add_done_callback(functools.partial(self.goal_response_callback, controller=ctrl, command_name=command_name, target=target, callback=callback))

    def goal_response_callback(self, future, controller, command_name, target, callback=None):
        """Handle action goal acceptance or rejection."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Command rejected: name={command_name}, controller={controller}, target={target}.")
            if callback: callback(_RejectedGoalResult())
            return
        self.get_logger().info(f"Command accepted: name={command_name}, controller={controller}, target={target}.")
        goal_handle.get_result_async().add_done_callback(functools.partial(self.get_result_callback, controller=controller, command_name=command_name, target=target, callback=callback))

    def feedback_callback(self, feedback_msg):
        """Log action execution feedback."""
        self.get_logger().info(f"Live action feedback received -> {feedback_msg.feedback}")

    def get_result_callback(self, future, controller, command_name, target, callback=None):
        """Log final action execution result and invoke completion callback."""
        result = future.result().result
        has_err = hasattr(result, "error_code")
        success = result.error_code == 0 if has_err else bool(result.reached_goal and not result.stalled)
        achieved = f"achieved_error_code={result.error_code}" if has_err else f"achieved_position={result.position}"
        self.get_logger().info(f"Command result: name={command_name}, controller={controller}, target={target}, {achieved}, success={success}.")
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
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
