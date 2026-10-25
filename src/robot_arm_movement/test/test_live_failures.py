from types import SimpleNamespace

import pytest

from robot_arm_movement.arm_controller_node import ComputRhythmArmController


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warn(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


def node(**values):
    logger = Logger()
    values.setdefault('logger', logger)
    values.setdefault('get_logger', lambda: logger)
    values.setdefault('pose_split_arm_gripper', 4)
    values.setdefault('runtime_ready', False)
    values.setdefault('arm_joint_names', None)
    values.setdefault('gripper_joint_name', None)
    return SimpleNamespace(**values)


def live_parameters(_name=None):
    return SimpleNamespace(get_parameter_value=lambda: SimpleNamespace(bool_value=False, double_value=15.0))


class Client:
    def __init__(self, available=True, service=True):
        self.available = available
        self.service = service

    def wait_for_server(self, timeout_sec):
        return self.available

    def wait_for_service(self, timeout_sec):
        return self.service


@pytest.mark.fault_injection
def test_missing_action_server_does_not_enable_runtime(monkeypatch):
    controller = node(
        get_parameter=live_parameters,
        _arm_action_client=Client(available=False),
        _gripper_action_client=Client(),
    )

    ComputRhythmArmController._discover_runtime(controller)

    assert controller.runtime_ready is False
    assert any('Action server availability' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_missing_parameter_service_does_not_enable_runtime():
    controller = node(
        get_parameter=live_parameters,
        _arm_action_client=Client(),
        _gripper_action_client=Client(),
        create_client=lambda service, name: Client(service=False),
    )

    ComputRhythmArmController._discover_runtime(controller)

    assert controller.runtime_ready is False
    assert any('parameter services are unavailable' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_parameter_timeout_does_not_enable_runtime(monkeypatch):
    future = SimpleNamespace(done=lambda: False)
    controller = node(
        get_parameter=live_parameters,
        _arm_action_client=Client(),
        _gripper_action_client=Client(),
        create_client=lambda service, name: Client(),
    )
    controller.create_client = lambda service, name: SimpleNamespace(wait_for_service=lambda timeout_sec: True, call_async=lambda request: future)
    monkeypatch.setattr('robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete', lambda *args, **kwargs: None)

    ComputRhythmArmController._discover_runtime(controller)

    assert controller.runtime_ready is False
    assert any('discovery timed out' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
@pytest.mark.parametrize('arm_joints, gripper_joint', [([], 'gripper_left_joint'), (['joint1', 'joint2', 'joint3'], 'gripper_left_joint'), (['joint1', 'joint2', 'joint3', 'joint4'], '')])
def test_invalid_discovered_joints_do_not_enable_runtime(monkeypatch, arm_joints, gripper_joint):
    arm_future = SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(values=[SimpleNamespace(string_array_value=arm_joints)]))
    gripper_future = SimpleNamespace(done=lambda: True, result=lambda: SimpleNamespace(values=[SimpleNamespace(string_value=gripper_joint)]))
    controller = node(
        get_parameter=live_parameters,
        _arm_action_client=Client(),
        _gripper_action_client=Client(),
        create_client=lambda service, name: SimpleNamespace(wait_for_service=lambda timeout_sec: True, call_async=lambda request: gripper_future if 'gripper' in name else arm_future),
    )
    monkeypatch.setattr('robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete', lambda *args, **kwargs: None)

    ComputRhythmArmController._discover_runtime(controller)

    assert controller.runtime_ready is False
    assert any('Invalid discovered joints' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_rejected_live_goal_is_reported():
    controller = node()
    future = SimpleNamespace(result=lambda: SimpleNamespace(accepted=False))

    ComputRhythmArmController.goal_response_callback(controller, future, 'gripper', 'cup', 0.015)

    assert any('Command rejected: name=cup, controller=gripper' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
@pytest.mark.parametrize('controller_name, result, expected', [
    ('arm', SimpleNamespace(error_code=-1), 'success=False'),
    ('gripper', SimpleNamespace(position=0.0, reached_goal=False, stalled=True), 'success=False'),
])
def test_unsuccessful_live_result_is_reported(controller_name, result, expected):
    controller = node()
    future = SimpleNamespace(result=lambda: SimpleNamespace(result=result))

    ComputRhythmArmController.get_result_callback(controller, future, controller_name, 'home', [0.0, -1.0, 0.0, 1.0] if controller_name == 'arm' else 0.015)

    assert any(expected in message for message in controller.logger.messages)
