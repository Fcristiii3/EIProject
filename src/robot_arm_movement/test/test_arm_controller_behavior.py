"""Unit tests for 4-DOF OpenMANIPULATOR-X kinematics, clamping, and game sequences."""

import math
from collections import namedtuple
import pytest
from robot_arm_movement.arm_controller_node import CLOSED_GRIPPER, DEFAULT_SPEC, GRAB_HOLD_POSE, GRAB_POSE, GRIPPER_MAX_RAD, GRIPPER_MIN_RAD, HOME_POSE, OPEN_GRIPPER, TAP_POSE, ComputRhythmArmController, build_grab_sequence, build_place_sequence, build_tap_sequence, clamp_gripper, clamp_pose, forward_kinematics_xyz, solve_open_manipulator_ik


class QuietLogger:
    """No-op logger stub."""

    def info(self, message):
        pass


Coordinates = namedtuple("Coordinates", ["x", "y", "z"])


class ResolutionController:
    """Lightweight controller double for pose and sequence planning tests."""

    __slots__ = ("available_choices", "home_pose", "tap_pose", "grab_pose", "grab_hold_pose", "spec", "holding_cup", "_held_cup_arm_pose", "param_arrays", "get_logger")

    def __init__(self):
        self.available_choices, self.home_pose, self.tap_pose, self.grab_pose, self.grab_hold_pose = ["tap", "grab"], list(HOME_POSE), list(TAP_POSE), list(GRAB_POSE), list(GRAB_HOLD_POSE)
        self.spec, self.holding_cup, self._held_cup_arm_pose, self.param_arrays, self.get_logger = DEFAULT_SPEC, False, None, {}, lambda: QuietLogger()

    def _safe_param_array(self, name, fallback):
        return list(self.param_arrays.get(name, fallback))

    def _safe_param_double(self, name, fallback):
        return float(fallback)

    def _safe_param_bool(self, name, fallback):
        return True if name == "auto_return_home" else bool(fallback)

    def _safe_param_str(self, name, fallback):
        return fallback

    def _choose_action(self):
        return ComputRhythmArmController._choose_action(self)

    def _resolve_target_pose(self, action_type, tempo_scale, msg_coords=None):
        return ComputRhythmArmController._resolve_target_pose(self, action_type, tempo_scale, msg_coords=msg_coords)


def test_reset_resolution_returns_home_pose(monkeypatch):
    controller = ResolutionController()
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.random.choice", lambda choices: "tap")
    assert ComputRhythmArmController._resolve_target_pose(controller, "reset", 1.0) == controller.home_pose


def test_interactive_tap_resolution_returns_tap_pose(monkeypatch):
    controller = ResolutionController()
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.random.choice", lambda choices: "tap")
    assert ComputRhythmArmController._resolve_target_pose(controller, "interactive", 1.0) == controller.tap_pose


def test_interactive_grab_resolution_returns_grab_pose(monkeypatch):
    controller = ResolutionController()
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.random.choice", lambda choices: "grab")
    assert ComputRhythmArmController._resolve_target_pose(controller, "interactive", 1.0) == controller.grab_pose


def test_gripper_limits_clamp_strictly_to_working_range():
    assert clamp_gripper(0.0) == pytest.approx(GRIPPER_MIN_RAD)
    assert clamp_gripper(GRIPPER_MIN_RAD - 0.5) == pytest.approx(GRIPPER_MIN_RAD)
    assert clamp_gripper(GRIPPER_MAX_RAD + 0.5) == pytest.approx(GRIPPER_MAX_RAD)
    assert clamp_gripper(math.radians(330.0)) == pytest.approx(math.radians(330.0))
    assert clamp_gripper(CLOSED_GRIPPER) == pytest.approx(GRIPPER_MIN_RAD)
    assert clamp_gripper(OPEN_GRIPPER) == pytest.approx(GRIPPER_MAX_RAD)
    clamped = clamp_pose([10.0, -5.0, 5.0, -5.0, 0.0])
    assert clamped[0] == pytest.approx(math.pi)
    assert clamped[1] == pytest.approx(-1.5)
    assert clamped[2] == pytest.approx(1.4)
    assert clamped[3] == pytest.approx(-1.7)
    assert clamped[4] == pytest.approx(GRIPPER_MIN_RAD)


@pytest.mark.parametrize("target", [(0.22, 0.0, 0.14), (0.18, 0.08, 0.10), (0.18, -0.08, 0.12), (0.25, 0.05, 0.16)])
def test_fk_ik_round_trip_accuracy(target):
    joints = solve_open_manipulator_ik(*target)
    assert len(joints) == 4
    for idx, val in enumerate(joints): assert DEFAULT_SPEC.arm_limits[idx][0] <= val <= DEFAULT_SPEC.arm_limits[idx][1]
    assert forward_kinematics_xyz(joints) == pytest.approx(target, abs=1e-3)


def test_out_of_workspace_ik_is_finite_and_clamped():
    joints = solve_open_manipulator_ik(5.0, -3.0, -2.0)
    assert all(math.isfinite(v) for v in joints)
    for idx, val in enumerate(joints): assert DEFAULT_SPEC.arm_limits[idx][0] <= val <= DEFAULT_SPEC.arm_limits[idx][1]


def test_camera_offset_and_scale_calibration_apply_before_ik(monkeypatch):
    controller = ResolutionController()
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.random.choice", lambda choices: "grab")
    controller.param_arrays = {"camera_offset_xyz": [0.0, 0.0, 0.12], "camera_scale_xyz": [1.0, 1.0, 1.0]}
    pose = ComputRhythmArmController._resolve_target_pose(controller, "camera", 1.0, Coordinates(x=0.20, y=0.05, z=0.0))
    assert pose[4] == pytest.approx(CLOSED_GRIPPER)
    assert forward_kinematics_xyz(pose[:4]) == pytest.approx((0.20, 0.05, 0.12), abs=1e-3)


def test_tap_grab_and_place_sequences_preserve_cup_rules():
    tap_steps = build_tap_sequence("cup", [0.1, -0.4, 0.2, 0.3], auto_sequence_enabled=True)
    assert [s.name for s in tap_steps] == ["cup", "cup_return"]
    assert tap_steps[0].pose[4] == pytest.approx(OPEN_GRIPPER)
    assert tap_steps[1].pose == pytest.approx(HOME_POSE)
    grab_steps = build_grab_sequence("cup", [0.2, -0.3, 0.2, 0.4], auto_sequence_enabled=True)
    assert [s.name for s in grab_steps] == ["cup_approach", "cup", "cup_lift", "cup_hold"]
    assert grab_steps[0].pose[4] == pytest.approx(OPEN_GRIPPER)
    assert grab_steps[1].pose[4] == pytest.approx(CLOSED_GRIPPER)
    assert grab_steps[2].pose[1] < grab_steps[1].pose[1]
    assert grab_steps[3].pose == pytest.approx(GRAB_HOLD_POSE)
    place_steps = build_place_sequence("place", [0.2, -0.3, 0.2, 0.4])
    assert [s.name for s in place_steps] == ["place_carry", "place_lower", "place_release", "place"]
    assert place_steps[0].pose[4] == pytest.approx(CLOSED_GRIPPER)
    assert place_steps[1].pose[:4] == pytest.approx([0.2, -0.3, 0.2, 0.4])
    assert place_steps[2].pose[4] == pytest.approx(OPEN_GRIPPER)
    assert place_steps[3].pose == pytest.approx(HOME_POSE)


def test_plan_action_remembers_and_places_held_cup(monkeypatch):
    controller = ResolutionController()
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.random.choice", lambda choices: "grab")
    steps, final_status, held_arm = ComputRhythmArmController._plan_action(controller, "camera", "camera", 1.0, Coordinates(x=0.22, y=0.0, z=0.14))
    assert final_status == "done:grab"
    assert len(steps) == 4
    controller.holding_cup, controller._held_cup_arm_pose = True, held_arm
    place_steps, place_status, next_held = ComputRhythmArmController._plan_action(controller, "reset", "place", 1.0)
    assert place_status == "done:place"
    assert next_held is None
    assert place_steps[1].pose[:4] == pytest.approx(held_arm)
    assert place_steps[-1].pose == pytest.approx(HOME_POSE)
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.random.choice", lambda choices: "tap")
    tap_steps, tap_status, tap_held = ComputRhythmArmController._plan_action(controller, "camera", "camera", 1.0, Coordinates(x=0.22, y=0.0, z=0.14))
    assert tap_status == "done:tap"
    assert tap_held is None
    assert tap_steps[0].pose[4] == pytest.approx(OPEN_GRIPPER)
