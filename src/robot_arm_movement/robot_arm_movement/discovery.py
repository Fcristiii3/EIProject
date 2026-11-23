"""Live ROS 2 discovery via /robot_state_publisher, /spawner_joint_state_broadcaster, and /joint_states."""

import math, time, xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import rclpy
from rcl_interfaces.srv import GetParameters
from robot_arm_movement.kinematics import DEFAULT_SPEC, RobotKinematicsSpec


@dataclass(frozen=True)
class DiscoveredRuntime:
    """Validated live robot runtime configuration discovered from ROS 2."""

    spec: RobotKinematicsSpec
    arm_joint_names: List[str]
    gripper_joint_name: str
    joint_states_received: bool
    observed_joint_names: List[str]


def _parse_xyz(text: Optional[str], fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if not text: return fallback
    parts = [float(v) for v in text.split()]
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else fallback


def _parse_limit(joint_elem: ET.Element, fallback: Tuple[float, float]) -> Tuple[float, float]:
    limit = joint_elem.find("limit")
    if limit is None: return fallback
    low = float(limit.attrib.get("lower", fallback[0]))
    high = float(limit.attrib.get("upper", fallback[1]))
    return (low, high) if low <= high else fallback


def _order_kinematic_chain(joints: List[dict]) -> List[dict]:
    if not joints: return []
    child_links = {j["child"] for j in joints}
    roots = [j for j in joints if j["parent"] not in child_links]
    by_parent: Dict[str, dict] = {j["parent"]: j for j in joints}
    ordered, current = [], roots[0] if roots else joints[0]
    seen = set()
    while current is not None and current["name"] not in seen:
        ordered.append(current)
        seen.add(current["name"])
        current = by_parent.get(current["child"])
    for j in joints:
        if j["name"] not in seen: ordered.append(j)
    return ordered


def parse_robot_description(urdf_xml: str, expected_arm_count: int = 4) -> RobotKinematicsSpec:
    """Parse /robot_state_publisher URDF XML into RobotKinematicsSpec without mimic or fixed joints."""
    if not urdf_xml or not urdf_xml.strip(): raise ValueError("Empty robot_description XML.")
    root = ET.fromstring(urdf_xml)
    arm_candidates, gripper_candidates = [], []
    origins_by_name: Dict[str, Tuple[float, float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        jtype = joint.attrib.get("type", "")
        origin_elem = joint.find("origin")
        xyz = _parse_xyz(origin_elem.attrib.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
        if name: origins_by_name[name] = xyz
        if jtype == "fixed" or joint.find("mimic") is not None: continue
        parent_elem, child_elem = joint.find("parent"), joint.find("child")
        parent = parent_elem.attrib.get("link", "") if parent_elem is not None else ""
        child = child_elem.attrib.get("link", "") if child_elem is not None else ""
        entry = {"name": name, "type": jtype, "parent": parent, "child": child, "origin": xyz, "limit": _parse_limit(joint, (-math.pi, math.pi) if jtype != "prismatic" else (-0.010, 0.019))}
        if jtype in ("revolute", "continuous"): arm_candidates.append(entry)
        elif jtype == "prismatic": gripper_candidates.append(entry)
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
    has_spawner = any("spawner_joint_state_broadcaster" in n for n in names)
    has_cm = any("controller_manager" in n for n in names)
    if has_spawner and not has_cm: return " /spawner_joint_state_broadcaster is active while /controller_manager is missing (check Dynamixel serial port/baudrate -3001/-3002 or fake.launch.py use_sim:=false)."
    if has_spawner: return " /spawner_joint_state_broadcaster is present; verify /controller_manager loaded joint_state_broadcaster, arm_controller, and gripper_controller."
    return ""


def _wait_for_joint_states(node, timeout_sec: float) -> Tuple[bool, List[str]]:
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while not getattr(node, "joint_states_received", False) and time.monotonic() < deadline:
        try: rclpy.spin_once(node, timeout_sec=0.1)
        except Exception: time.sleep(0.05)
    received = bool(getattr(node, "joint_states_received", False))
    observed = list(getattr(node, "latest_joint_names", []))
    return received, observed


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
    arm_ready = arm_action_client.wait_for_server(timeout_sec=timeout_sec)
    gripper_ready = gripper_action_client.wait_for_server(timeout_sec=timeout_sec)
    if not arm_ready or not gripper_ready: return None, f"Action server availability: arm={arm_ready}, gripper={gripper_ready}.{_spawner_diagnostic(node)}"
    received, observed = _wait_for_joint_states(node, timeout_sec=min(timeout_sec, 5.0))
    if not received: return None, f"Timed out waiting for {joint_states_topic} from joint_state_broadcaster.{_spawner_diagnostic(node)}"
    required = list(spec.arm_joint_names) + [spec.gripper_joint_name]
    missing = [name for name in required if name not in observed]
    if missing: return None, f"Joint mismatch between /robot_state_publisher and {joint_states_topic}: missing={missing}, observed={observed}."
    return DiscoveredRuntime(spec=spec, arm_joint_names=list(spec.arm_joint_names), gripper_joint_name=spec.gripper_joint_name, joint_states_received=True, observed_joint_names=observed), None
