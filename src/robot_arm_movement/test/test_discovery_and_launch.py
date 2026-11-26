"""Tests for URDF discovery, /joint_states & /comput_game topics, and launch description."""

import importlib.util, math
from pathlib import Path
import pytest
import rclpy
from geometry_msgs.msg import Point
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch_ros.utilities import evaluate_parameters
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from robot_arm_movement.arm_controller_node import CLOSED_GRIPPER, GRIPPER_MAX_RAD, GRIPPER_MIN_RAD, OPEN_GRIPPER, ComputRhythmArmController, parse_robot_description

SAMPLE_URDF = """<?xml version="1.0"?>
<robot name="open_manipulator_x">
  <link name="world"/>
  <joint name="world_fixed" type="fixed"><parent link="world"/><child link="link1"/><origin xyz="0 0 0" rpy="0 0 0"/></joint>
  <link name="link1"/>
  <joint name="joint1" type="revolute"><parent link="link1"/><child link="link2"/><origin xyz="0.012 0.0 0.017" rpy="0 0 0"/><limit lower="-3.14159" upper="3.14159" effort="1" velocity="4.8"/></joint>
  <link name="link2"/>
  <joint name="joint2" type="revolute"><parent link="link2"/><child link="link3"/><origin xyz="0.0 0.0 0.0595" rpy="0 0 0"/><limit lower="-1.5" upper="1.5" effort="1" velocity="4.8"/></joint>
  <link name="link3"/>
  <joint name="joint3" type="revolute"><parent link="link3"/><child link="link4"/><origin xyz="0.024 0.0 0.128" rpy="0 0 0"/><limit lower="-1.5" upper="1.4" effort="1" velocity="4.8"/></joint>
  <link name="link4"/>
  <joint name="joint4" type="revolute"><parent link="link4"/><child link="link5"/><origin xyz="0.124 0.0 0.0" rpy="0 0 0"/><limit lower="-1.7" upper="1.97" effort="1" velocity="4.8"/></joint>
  <link name="link5"/>
  <joint name="gripper_left_joint" type="prismatic"><parent link="link5"/><child link="gripper_left_link"/><origin xyz="0.0817 0.021 0.0" rpy="0 0 0"/><limit lower="-0.010" upper="0.019" effort="1" velocity="4.8"/></joint>
  <link name="gripper_left_link"/>
  <joint name="gripper_right_joint" type="prismatic"><parent link="link5"/><child link="gripper_right_link"/><limit lower="-0.010" upper="0.019" effort="1" velocity="4.8"/><mimic joint="gripper_left_joint" multiplier="1"/></joint>
  <link name="gripper_right_link"/>
  <joint name="end_effector_joint" type="fixed"><parent link="link5"/><child link="end_effector_link"/><origin xyz="0.126 0.0 0.0" rpy="0 0 0"/></joint>
  <link name="end_effector_link"/>
</robot>
"""


def _spin_until(node, cond):
    for _ in range(40):
        if cond(): return
        rclpy.spin_once(node, timeout_sec=0.05)


def test_parse_robot_description_excludes_mimic_and_orders_chain():
    spec = parse_robot_description(SAMPLE_URDF)
    assert spec.arm_joint_names == ("joint1", "joint2", "joint3", "joint4")
    assert spec.gripper_joint_name == "gripper_left_joint"
    assert spec.gripper_limits == pytest.approx((GRIPPER_MIN_RAD, GRIPPER_MAX_RAD))
    assert spec.joint3_origin == pytest.approx((0.024, 0.0, 0.128))
    assert spec.end_effector_origin == pytest.approx((0.126, 0.0, 0.0))
    assert spec.arm_limits[0][0] == pytest.approx(-math.pi, abs=1e-4)
    sub_urdf = SAMPLE_URDF.replace('lower="-0.010" upper="0.019"', 'lower="-0.042" upper="-0.030"')
    assert parse_robot_description(sub_urdf).gripper_limits == pytest.approx((-0.042, -0.030))


@pytest.mark.parametrize("bad_urdf", ["", "<robot/>", "<robot><joint name='j1' type='revolute'><parent link='a'/><child link='b'/></joint></robot>"])
def test_parse_robot_description_rejects_invalid_urdf(bad_urdf):
    with pytest.raises(ValueError): parse_robot_description(bad_urdf)


def test_mock_node_tracks_joint_states_tempo_and_status_sequences():
    rclpy.init(args=["--ros-args", "-p", "mock_mode_enabled:=true"])
    node = ComputRhythmArmController()
    statuses = []
    sub = node.create_subscription(String, "/comput_game/arm_status", lambda msg: statuses.append(msg.data), 10)
    try:
        node.set_parameters([rclpy.parameter.Parameter("mock_execution_delay", rclpy.parameter.Parameter.Type.DOUBLE, 0.05), rclpy.parameter.Parameter("auto_return_home", rclpy.parameter.Parameter.Type.BOOL, True), rclpy.parameter.Parameter("camera_action_mode", rclpy.parameter.Parameter.Type.STRING, "grab")])
        _spin_until(node, lambda: "ready" in statuses)
        assert "ready" in statuses
        node.joint_state_callback(JointState(name=["joint1", "joint2", "joint3", "joint4", "gripper_left_joint"], position=[0.1, -0.2, 0.3, 0.4, OPEN_GRIPPER]))
        assert node.joint_states_received is True
        assert node.latest_joint_positions["gripper_left_joint"] == pytest.approx(OPEN_GRIPPER)
        node.tempo_callback(Float64(data=0.5))
        assert node._get_tempo_scale() == pytest.approx(0.5)
        node.camera_coord_callback(Point(x=0.20, y=0.0, z=0.14))
        _spin_until(node, lambda: "done:grab" in statuses)
        assert "done:grab" in statuses
        assert node.holding_cup is True
        node.game_action_callback(String(data="place"))
        _spin_until(node, lambda: "done:place" in statuses and statuses[-1] == "ready")
        assert "done:place" in statuses
        assert node.holding_cup is False
    finally:
        node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("mode, expected_flag", [("mock", True), ("fake", False), ("attach", False), ("hardware", False), ("gazebo", False)])
def test_launch_file_supports_all_modes(mode, expected_flag):
    launch_path = Path(__file__).parents[1] / "launch" / "robot_arm_movement.launch.py"
    spec = importlib.util.spec_from_file_location("robot_arm_launch", launch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ld = mod.generate_launch_description()
    ctx = LaunchContext()
    for entity in ld.entities:
        if isinstance(entity, DeclareLaunchArgument): entity.execute(ctx)
    ctx.launch_configurations["mode"] = mode
    ctx.launch_configurations["launch_camera"] = "true"
    actions = mod._launch_setup(ctx)
    arm_nodes = [a for a in actions if getattr(a, "_Node__package", None) == "robot_arm_movement"]
    cam_nodes = [a for a in actions if getattr(a, "_Node__package", None) == "camera"]
    assert len(arm_nodes) == 1
    assert len(cam_nodes) == 1
    eval_params = evaluate_parameters(ctx, arm_nodes[0]._Node__parameters)
    assert eval_params[0]["mock_mode_enabled"] is expected_flag
    assert eval_params[0]["use_sim_time"] is (mode == "gazebo")
    assert eval_params[0]["gripper_closed_position"] == pytest.approx(CLOSED_GRIPPER)
    assert eval_params[0]["gripper_open_position"] == pytest.approx(OPEN_GRIPPER)
    if mode == "fake":
        packages = [getattr(a, "_Node__package", None) for a in actions]
        assert "robot_state_publisher" in packages
        assert "controller_manager" in packages
