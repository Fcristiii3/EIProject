"""Analytical 4-DOF OpenMANIPULATOR-X kinematics, pose clamping, and game motion sequences."""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

OPEN_GRIPPER = 0.015
CLOSED_GRIPPER = -0.010
HOME_POSE = [0.0, -1.0, 0.0, 1.0, OPEN_GRIPPER]
TAP_POSE = [0.1, -0.4, 0.2, 0.3, OPEN_GRIPPER]
GRAB_POSE = [0.1, -0.4, 0.2, 0.3, CLOSED_GRIPPER]
GRAB_HOLD_POSE = [1.45, -0.9, 0.2, 0.7, CLOSED_GRIPPER]


@dataclass(frozen=True)
class RobotKinematicsSpec:
    """URDF-derived link offsets, joint names, and physical joint limits."""

    arm_joint_names: Tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4")
    gripper_joint_name: str = "gripper_left_joint"
    arm_limits: Tuple[Tuple[float, float], ...] = ((-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97))
    gripper_limits: Tuple[float, float] = (-0.010, 0.019)
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


def clamp(value: float, low: float, high: float) -> float:
    """Clamp numeric value to inclusive [low, high] range."""
    return max(float(low), min(float(high), float(value)))


def clamp_gripper(value: float, spec: RobotKinematicsSpec = DEFAULT_SPEC) -> float:
    """Clamp gripper position strictly within URDF prismatic limits."""
    return clamp(value, spec.gripper_limits[0], spec.gripper_limits[1])


def clamp_arm_joints(joints: Sequence[float], spec: RobotKinematicsSpec = DEFAULT_SPEC) -> List[float]:
    """Clamp all four arm joints strictly within URDF revolute limits."""
    return [clamp(joints[i], spec.arm_limits[i][0], spec.arm_limits[i][1]) for i in range(4)]


def clamp_pose(pose: Sequence[float], spec: RobotKinematicsSpec = DEFAULT_SPEC) -> List[float]:
    """Clamp a 5-element [joint1..joint4, gripper] pose to physical URDF limits."""
    return clamp_arm_joints(pose[:4], spec) + [clamp_gripper(pose[4], spec)]


def forward_kinematics_xyz(arm_joints: Sequence[float], spec: RobotKinematicsSpec = DEFAULT_SPEC) -> Tuple[float, float, float]:
    """Compute end_effector_link (x, y, z) in link1 frame for 4 arm joint angles."""
    q1, q2, q3, q4 = (float(arm_joints[i]) for i in range(4))
    x1, y1, z1 = spec.joint1_origin
    x2, y2, z2 = spec.joint2_origin
    x3, _, z3 = spec.joint3_origin
    x4, _, z4 = spec.joint4_origin
    xe, _, ze = spec.end_effector_origin
    p3, p4 = q2 + q3, q2 + q3 + q4
    r = x2 + (x3 * math.cos(q2) + z3 * math.sin(q2)) + (x4 * math.cos(p3) + z4 * math.sin(p3)) + (xe * math.cos(p4) + ze * math.sin(p4))
    z = z1 + z2 + (-x3 * math.sin(q2) + z3 * math.cos(q2)) + (-x4 * math.sin(p3) + z4 * math.cos(p3)) + (-xe * math.sin(p4) + ze * math.cos(p4))
    return (x1 + r * math.cos(q1) - y2 * math.sin(q1), y1 + r * math.sin(q1) + y2 * math.cos(q1), z)


def solve_open_manipulator_ik(x: float, y: float, z: float, spec: RobotKinematicsSpec = DEFAULT_SPEC, preferred_pitch: float = 0.3, camera_offset_xyz: Sequence[float] = (0.0, 0.0, 0.0), camera_scale_xyz: Sequence[float] = (1.0, 1.0, 1.0)) -> List[float]:
    """Solve analytical 4-DOF OpenMANIPULATOR-X IK with radial clamping and pitch fallback."""
    tx = float(x) * float(camera_scale_xyz[0]) + float(camera_offset_xyz[0])
    ty = float(y) * float(camera_scale_xyz[1]) + float(camera_offset_xyz[1])
    tz = float(z) * float(camera_scale_xyz[2]) + float(camera_offset_xyz[2])
    x1, y1, z1 = spec.joint1_origin
    x2, _, z2 = spec.joint2_origin
    x3, _, z3 = spec.joint3_origin
    x4, _, z4 = spec.joint4_origin
    xe, _, ze = spec.end_effector_origin
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
        rw = r_target - (xe * math.cos(phi) + ze * math.sin(phi))
        zw = z_target - (-xe * math.sin(phi) + ze * math.cos(phi))
        d = math.hypot(rw, zw)
        if d < 1e-6: rw, zw, d = 0.0, d_min, d_min
        elif d < d_min or d > d_max: scale = clamp(d, d_min, d_max) / d; rw *= scale; zw *= scale; d = clamp(d, d_min, d_max)
        cos_delta = clamp((d * d - l2 * l2 - l3 * l3) / (2.0 * l2 * l3), -1.0, 1.0)
        gamma = math.atan2(rw, zw)
        for elbow_sign in (1.0, -1.0):
            delta = elbow_sign * math.acos(cos_delta)
            psi = math.atan2(l3 * math.sin(delta), l2 + l3 * math.cos(delta))
            q2 = (gamma - psi) - alpha
            q3 = delta - (math.pi / 2.0) + alpha + beta
            q4 = phi - q2 - q3
            raw = [q1_raw, q2, q3, q4]
            clamped = clamp_arm_joints(raw, spec)
            fx, fy, fz = forward_kinematics_xyz(clamped, spec)
            err = math.sqrt((fx - tx) ** 2 + (fy - ty) ** 2 + (fz - tz) ** 2)
            in_bounds = all(spec.arm_limits[i][0] - 1e-6 <= raw[i] <= spec.arm_limits[i][1] + 1e-6 for i in range(4))
            if in_bounds and err < 1e-3: return clamped
            score = (round(err, 6), abs(phi - preferred_pitch))
            if score < best_score: best_score, best_pose = score, clamped
    return best_pose if best_pose is not None else clamp_arm_joints([q1_raw, -0.4, 0.2, 0.3], spec)


def _lift_pose(arm_pose: Sequence[float], gripper_pos: float, spec: RobotKinematicsSpec) -> List[float]:
    return clamp_arm_joints([arm_pose[0], -0.8, 0.1, 0.7], spec) + [clamp_gripper(gripper_pos, spec)]


def build_tap_sequence(cmd: str, arm_pose: Sequence[float], home_pose: Sequence[float] = HOME_POSE, spec: RobotKinematicsSpec = DEFAULT_SPEC, auto_sequence_enabled: bool = True) -> List[MotionStep]:
    """Build single-step or auto-retracting tap sequence keeping the cup on the table."""
    open_pos = clamp_gripper(OPEN_GRIPPER, spec)
    tap_step = MotionStep(cmd, clamp_arm_joints(arm_pose, spec) + [open_pos], 1.0)
    if not auto_sequence_enabled: return [tap_step]
    return [tap_step, MotionStep(f"{cmd}_return", clamp_pose(home_pose, spec), 1.0)]


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
