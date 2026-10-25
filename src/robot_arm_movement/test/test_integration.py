import os
import signal
import subprocess
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String


COMMANDS = ['interact', 'cup', 'box', 'cloth', 'reset', 'home']


def wait_for_subscription(node, publisher, executor, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if publisher.get_subscription_count() > 0 or node.count_subscribers(publisher.topic_name) > 0:
            return
        executor.spin_once(timeout_sec=0.2)
        time.sleep(0.2)
    raise AssertionError('No subscriber became available for the integration-test publisher.')


@pytest.fixture
def ros_publisher():
    rclpy.init()
    node = rclpy.create_node('robot_arm_integration_test_publisher')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    yield node, executor
    node.destroy_node()
    rclpy.shutdown()


def publish_command(node, executor, command, timeout):
    publisher = node.create_publisher(String, '/comput_game/target_action', 10)
    try:
        wait_for_subscription(node, publisher, executor, timeout)
        message = String()
        message.data = command
        for _ in range(3):
            publisher.publish(message)
            executor.spin_once(timeout_sec=0.2)
            time.sleep(0.2)
    finally:
        node.destroy_publisher(publisher)


def publish_camera_coordinates(node, executor, timeout):
    publisher = node.create_publisher(Point, '/comput_game/camera_coordinates', 10)
    try:
        wait_for_subscription(node, publisher, executor, timeout)
        message = Point(x=0.10, y=0.20, z=0.30)
        for _ in range(3):
            publisher.publish(message)
            executor.spin_once(timeout_sec=0.2)
            time.sleep(0.2)
    finally:
        node.destroy_publisher(publisher)


def wait_for_log(log_path, text, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = Path(log_path).read_text(errors='replace')
        if text in output:
            return output
        time.sleep(0.05)
    raise AssertionError(f'Log text not observed: {text}\n{output[-4000:]}')


@pytest.fixture
def controller_process(request):
    mode = request.config.getoption('--robot-arm-mode')
    timeout = request.config.getoption('--robot-arm-timeout')
    log_file = NamedTemporaryFile(prefix='robot_arm_integration_', suffix='.log', delete=False)
    process = subprocess.Popen(['ros2', 'run', 'robot_arm_movement', 'arm_controller_node', '--ros-args', '-p', f'mock_mode_enabled:={mode == "mock"}'], stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_for_log(log_file.name, 'Ready for rhythm game commands', timeout=timeout)
        yield process, log_file.name, timeout, mode
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)
        log_file.close()


@pytest.mark.integration
def test_all_command_rules_and_camera_input(controller_process, ros_publisher):
    process, log_path, timeout, mode = controller_process
    node, executor = ros_publisher
    for command in COMMANDS:
        publish_command(node, executor, command, timeout)
        wait_for_log(log_path, f'{"command" if mode == "mock" else "name"}={command}', timeout=timeout)
    publish_camera_coordinates(node, executor, timeout)
    wait_for_log(log_path, f'{"command" if mode == "mock" else "name"}=camera', timeout=timeout)
    output = Path(log_path).read_text(errors='replace')
    if mode == 'mock':
        assert output.count('Mock arm goal completed') >= len(COMMANDS) + 1
        assert output.count('Mock gripper goal completed') >= len(COMMANDS) + 1
        assert 'Ready for rhythm game commands' in output
    else:
        assert output.count('controller=arm') >= len(COMMANDS) + 1
        assert output.count('controller=gripper') >= len(COMMANDS) + 1
        assert output.count('success=True') >= len(COMMANDS) + 1
    assert 'mismatch' not in output.lower()


@pytest.mark.integration
def test_reset_and_home_dispatch_distinct_named_commands(controller_process, ros_publisher):
    process, log_path, timeout, mode = controller_process
    node, executor = ros_publisher
    publish_command(node, executor, 'reset', timeout)
    reset_output = wait_for_log(log_path, f'{"command" if mode == "mock" else "name"}=reset', timeout=timeout)
    publish_command(node, executor, 'home', timeout)
    home_output = wait_for_log(log_path, f'{"command" if mode == "mock" else "name"}=home', timeout=timeout)
    if mode == 'mock':
        assert 'Mock arm goal completed: command=reset, target=[0.0, -1.0, 0.0, 1.0]' in reset_output
        assert 'Mock gripper goal completed: command=reset, target=0.015' in reset_output
        assert 'Mock arm goal completed: command=home, target=[0.0, -1.0, 0.0, 1.0]' in home_output
        assert 'Mock gripper goal completed: command=home, target=0.015' in home_output
    else:
        output = Path(log_path).read_text(errors='replace')
        assert 'Command result: name=reset, controller=arm' in output
        assert 'Command result: name=reset, controller=gripper' in output
        assert 'Command result: name=home, controller=arm' in output
        assert 'Command result: name=home, controller=gripper' in output
