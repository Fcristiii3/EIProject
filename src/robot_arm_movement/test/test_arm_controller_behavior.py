from types import SimpleNamespace

import pytest

from robot_arm_movement.arm_controller_node import ComputRhythmArmController


class QuietLogger:
    def info(self, message):
        pass


def controller_for_resolution():
    return SimpleNamespace(
        available_choices=['tap', 'grab'],
        home_pose=[0.0, -1.0, 0.0, 1.0, 0.015],
        tap_pose=[0.1, -0.4, 0.2, 0.3, 0.015],
        grab_pose=[0.1, -0.4, 0.2, 0.3, -0.015],
        get_logger=lambda: QuietLogger(),
    )


def test_reset_resolution_returns_home_pose(monkeypatch):
    controller = controller_for_resolution()
    monkeypatch.setattr('robot_arm_movement.arm_controller_node.random.choice', lambda choices: 'tap')

    assert ComputRhythmArmController._resolve_target_pose(controller, 'reset', 1.0) == controller.home_pose


def test_interactive_tap_resolution_returns_tap_pose(monkeypatch):
    controller = controller_for_resolution()
    monkeypatch.setattr('robot_arm_movement.arm_controller_node.random.choice', lambda choices: 'tap')

    assert ComputRhythmArmController._resolve_target_pose(controller, 'interactive', 1.0) == controller.tap_pose


def test_interactive_grab_resolution_returns_grab_pose(monkeypatch):
    controller = controller_for_resolution()
    monkeypatch.setattr('robot_arm_movement.arm_controller_node.random.choice', lambda choices: 'grab')

    assert ComputRhythmArmController._resolve_target_pose(controller, 'interactive', 1.0) == controller.grab_pose


def test_camera_resolution_translates_coordinates_and_preserves_choice(monkeypatch):
    controller = controller_for_resolution()
    coordinates = SimpleNamespace(x=0.2, y=0.7, z=0.4)
    monkeypatch.setattr('robot_arm_movement.arm_controller_node.random.choice', lambda choices: 'grab')

    assert ComputRhythmArmController._resolve_target_pose(controller, 'camera', 1.0, coordinates) == pytest.approx([0.2, 0.2, 0.6, 0.3, -0.015])


def test_pose_split_has_four_arm_values_and_one_gripper_value():
    pose = [0.0, -1.0, 0.0, 1.0, 0.015]

    assert pose[:4] == [0.0, -1.0, 0.0, 1.0]
    assert pose[4] == 0.015
