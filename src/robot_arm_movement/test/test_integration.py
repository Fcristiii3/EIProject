"""End-to-end ROS 2 integration tests across mock, fake ros2_control, and live modes."""

import os, signal, subprocess, time
from pathlib import Path
from tempfile import NamedTemporaryFile
import pytest
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import Float64, String
from robot_arm_movement.arm_controller_node import OPEN_GRIPPER

COMMANDS = ["interact", "cup", "box", "place", "cloth", "reset", "home"]


def wait_for_subscription(node, publisher, executor, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if publisher.get_subscription_count() > 0 or node.count_subscribers(publisher.topic_name) > 0: return
        executor.spin_once(timeout_sec=0.2); time.sleep(0.1)
    raise AssertionError(f"No subscriber became available for {publisher.topic_name}.")


@pytest.fixture
def ros_publisher():
    rclpy.init()
    node = rclpy.create_node("robot_arm_integration_test_publisher")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    yield node, executor
    node.destroy_node()
    rclpy.shutdown()


def publish_msg(node, executor, msg_type, topic, msg, timeout):
    publisher = node.create_publisher(msg_type, topic, 10)
    try:
        wait_for_subscription(node, publisher, executor, timeout)
        publisher.publish(msg)
        executor.spin_once(timeout_sec=0.2); time.sleep(0.15)
    finally: node.destroy_publisher(publisher)


def wait_for_log(log_path, text, timeout=15.0):
    deadline, output = time.monotonic() + timeout, ""
    while time.monotonic() < deadline:
        output = Path(log_path).read_text(errors="replace")
        if text in output: return output
        time.sleep(0.05)
    raise AssertionError(f"Log text not observed: {text}\n{output[-4000:]}")


@pytest.fixture
def controller_process(request):
    mode, timeout = request.config.getoption("--robot-arm-mode"), request.config.getoption("--robot-arm-timeout")
    log_file = NamedTemporaryFile(prefix="robot_arm_integration_", suffix=".log", delete=False)
    if mode == "fake": cmd = ["ros2", "launch", "robot_arm_movement", "robot_arm_movement.launch.py", "mode:=fake", "auto_return_home:=true", "tempo_scale:=0.4", f"controller_discovery_timeout_sec:={timeout}"]
    else: cmd = ["ros2", "run", "robot_arm_movement", "arm_controller_node", "--ros-args", "-p", f"mock_mode_enabled:={mode == 'mock'}", "-p", "auto_return_home:=true", "-p", "tempo_scale:=0.4", "-p", "mock_execution_delay:=0.05", "-p", f"controller_discovery_timeout_sec:={timeout}"]
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_for_log(log_file.name, "Ready for rhythm game commands", timeout=timeout)
        yield process, log_file.name, timeout, mode
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try: process.wait(timeout=10.0)
        except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5.0)
        log_file.close()


@pytest.mark.integration
def test_all_command_rules_and_camera_input(controller_process, ros_publisher):
    _, log_path, timeout, mode = controller_process
    node, executor = ros_publisher
    statuses = []
    status_sub = node.create_subscription(String, "/comput_game/arm_status", lambda msg: statuses.append(msg.data), 10)
    try:
        publish_msg(node, executor, Float64, "/comput_game/tempo", Float64(data=0.5), timeout)
        wait_for_log(log_path, "tempo_scale=0.500", timeout=timeout)
        for command in COMMANDS:
            publish_msg(node, executor, String, "/comput_game/target_action", String(data=command), timeout)
            wait_for_log(log_path, f"{'command' if mode == 'mock' else 'name'}={command}", timeout=timeout)
        publish_msg(node, executor, Point, "/comput_game/camera_coordinates", Point(x=0.20, y=0.05, z=0.14), timeout)
        wait_for_log(log_path, f"{'command' if mode == 'mock' else 'name'}=camera", timeout=timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not any(s in ("done:tap", "done:grab") for s in statuses): executor.spin_once(timeout_sec=0.1)
        output = Path(log_path).read_text(errors="replace")
        if mode == "mock":
            assert output.count("Mock arm goal completed") >= len(COMMANDS) + 1
            assert output.count("Mock gripper goal completed") >= len(COMMANDS) + 1
        else:
            assert output.count("controller=arm") >= len(COMMANDS) + 1
            assert output.count("controller=gripper") >= len(COMMANDS) + 1
            assert output.count("success=True") >= len(COMMANDS) + 1
        assert "mismatch" not in output.lower()
    finally: node.destroy_subscription(status_sub)


@pytest.mark.integration
def test_reset_and_home_dispatch_distinct_named_commands(controller_process, ros_publisher):
    _, log_path, timeout, mode = controller_process
    node, executor = ros_publisher
    publish_msg(node, executor, String, "/comput_game/target_action", String(data="reset"), timeout)
    reset_output = wait_for_log(log_path, f"{'command' if mode == 'mock' else 'name'}=reset", timeout=timeout)
    publish_msg(node, executor, String, "/comput_game/target_action", String(data="home"), timeout)
    home_output = wait_for_log(log_path, f"{'command' if mode == 'mock' else 'name'}=home", timeout=timeout)
    if mode == "mock":
        assert "Mock arm goal completed: command=reset, target=[0.0, -1.0, 0.0, 1.0]" in reset_output
        assert f"Mock gripper goal completed: command=reset, target={OPEN_GRIPPER}" in reset_output
        assert "Mock arm goal completed: command=home, target=[0.0, -1.0, 0.0, 1.0]" in home_output
        assert f"Mock gripper goal completed: command=home, target={OPEN_GRIPPER}" in home_output
    else:
        output = Path(log_path).read_text(errors="replace")
        assert "Command result: name=reset, controller=arm" in output
        assert "Command result: name=reset, controller=gripper" in output
        assert "Command result: name=home, controller=arm" in output
        assert "Command result: name=home, controller=gripper" in output
