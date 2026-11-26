"""Deterministic fault-injection tests for /robot_state_publisher, /spawner_joint_state_broadcaster, and live discovery."""

from collections import namedtuple
import pytest
from robot_arm_movement.arm_controller_node import OPEN_GRIPPER, ComputRhythmArmController, discover_live_runtime

VALID_URDF = """<robot name="omx">
  <link name="l1"/><joint name="joint1" type="revolute"><parent link="l1"/><child link="l2"/><limit lower="-3.14" upper="3.14"/></joint>
  <link name="l2"/><joint name="joint2" type="revolute"><parent link="l2"/><child link="l3"/><limit lower="-1.5" upper="1.5"/></joint>
  <link name="l3"/><joint name="joint3" type="revolute"><parent link="l3"/><child link="l4"/><limit lower="-1.5" upper="1.4"/></joint>
  <link name="l4"/><joint name="joint4" type="revolute"><parent link="l4"/><child link="l5"/><limit lower="-1.7" upper="1.97"/></joint>
  <link name="l5"/><joint name="gripper_left_joint" type="prismatic"><parent link="l5"/><child link="gl"/><limit lower="-0.010" upper="0.019"/></joint>
  <link name="gl"/>
</robot>"""


class Logger:
    """In-memory logger stub."""

    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warn(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


class Client:
    """Action/service availability stub."""

    __slots__ = ("available", "service", "future")

    def __init__(self, available=True, service=True, future=None):
        self.available, self.service, self.future = available, service, future

    def wait_for_server(self, timeout_sec):
        return self.available

    def wait_for_service(self, timeout_sec):
        return self.service

    def call_async(self, request):
        return self.future


class ActionFuture:
    """Future stub holding a pre-baked response."""

    __slots__ = ("_response",)

    def __init__(self, response=None):
        self._response = response

    def done(self):
        return self._response is not None

    def result(self):
        return self._response


ParameterValue = namedtuple("ParameterValue", ["bool_value", "double_value"])


class LiveParameter:
    """Parameter value stub for live discovery tests."""

    __slots__ = ()

    def get_parameter_value(self):
        return ParameterValue(bool_value=False, double_value=0.05)


GoalHandle = namedtuple("GoalHandle", ["accepted"])
ResultResponse = namedtuple("ResultResponse", ["result"])
ArmResult = namedtuple("ArmResult", ["error_code"])
GripperResult = namedtuple("GripperResult", ["position", "reached_goal", "stalled"])
StringValue = namedtuple("StringValue", ["string_value"])
DiscoveredResponse = namedtuple("DiscoveredResponse", ["values"])


class Controller:
    """Controller fake for /robot_state_publisher and /joint_states discovery failure tests."""

    __slots__ = ("_arm_action_client", "_gripper_action_client", "create_client", "get_parameter", "logger", "pose_split_arm_gripper", "runtime_ready", "spec", "arm_joint_names", "gripper_joint_name", "joint_states_received", "latest_joint_names", "node_names", "statuses")

    def __init__(self):
        self.logger, self.get_parameter = Logger(), lambda _name=None: LiveParameter()
        self.create_client = lambda service, name: Client(future=ActionFuture(DiscoveredResponse(values=[StringValue(string_value=VALID_URDF)])))
        self.pose_split_arm_gripper, self.runtime_ready, self.spec, self.arm_joint_names, self.gripper_joint_name = 4, False, None, None, None
        self.joint_states_received, self.latest_joint_names = True, ["joint1", "joint2", "joint3", "joint4", "gripper_left_joint"]
        self.node_names, self.statuses, self._arm_action_client, self._gripper_action_client = [], [], Client(), Client()

    def get_logger(self):
        return self.logger

    def get_node_names(self):
        return list(self.node_names)

    def _publish_status(self, status):
        self.statuses.append(status)


def node(**values):
    controller = Controller()
    for key, value in values.items(): setattr(controller, key, value)
    return controller


@pytest.mark.fault_injection
def test_missing_robot_state_publisher_does_not_enable_runtime():
    controller = node(create_client=lambda service, name: Client(service=False), node_names=["/spawner_joint_state_broadcaster"])
    ComputRhythmArmController._discover_runtime(controller)
    assert controller.runtime_ready is False
    assert any("/robot_state_publisher/get_parameters" in msg for msg in controller.logger.messages)
    assert any("/spawner_joint_state_broadcaster" in msg for msg in controller.logger.messages)


@pytest.mark.fault_injection
def test_robot_state_publisher_timeout_does_not_enable_runtime(monkeypatch):
    controller = node(create_client=lambda service, name: Client(future=ActionFuture(None)))
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete", lambda *args, **kwargs: None)
    ComputRhythmArmController._discover_runtime(controller)
    assert controller.runtime_ready is False
    assert any("timed out" in msg for msg in controller.logger.messages)


@pytest.mark.fault_injection
@pytest.mark.parametrize("bad_urdf", ["", "<robot><joint name='joint1' type='revolute'><parent link='a'/><child link='b'/></joint></robot>"])
def test_invalid_urdf_does_not_enable_runtime(monkeypatch, bad_urdf):
    controller = node(create_client=lambda service, name: Client(future=ActionFuture(DiscoveredResponse(values=[StringValue(string_value=bad_urdf)]))))
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete", lambda *args, **kwargs: None)
    ComputRhythmArmController._discover_runtime(controller)
    assert controller.runtime_ready is False
    assert any("Invalid robot_description" in msg for msg in controller.logger.messages)


@pytest.mark.fault_injection
def test_stuck_spawner_reports_missing_controller_manager(monkeypatch):
    controller = node(_arm_action_client=Client(available=False), node_names=["/robot_state_publisher", "/spawner_joint_state_broadcaster"])
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete", lambda *args, **kwargs: None)
    runtime, err = discover_live_runtime(controller, controller._arm_action_client, controller._gripper_action_client, timeout_sec=0.05)
    assert runtime is None
    assert "Action server availability" in err
    assert "/spawner_joint_state_broadcaster is active while /controller_manager is missing" in err


@pytest.mark.fault_injection
def test_missing_joint_states_timeout_fails_discovery(monkeypatch):
    controller = node(joint_states_received=False, latest_joint_names=[])
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete", lambda *args, **kwargs: None)
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_once", lambda *args, **kwargs: None)
    runtime, err = discover_live_runtime(controller, controller._arm_action_client, controller._gripper_action_client, timeout_sec=0.05)
    assert runtime is None
    assert "Timed out waiting for /joint_states" in err


@pytest.mark.fault_injection
def test_joint_states_mismatch_fails_discovery(monkeypatch):
    controller = node(joint_states_received=True, latest_joint_names=["joint1", "joint2", "joint3", "joint4"])
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete", lambda *args, **kwargs: None)
    ComputRhythmArmController._discover_runtime(controller)
    assert controller.runtime_ready is False
    assert any("Joint mismatch" in msg and "gripper_left_joint" in msg for msg in controller.logger.messages)


@pytest.mark.fault_injection
def test_rejected_live_goal_is_reported():
    controller = node()
    future = ActionFuture(GoalHandle(accepted=False))
    ComputRhythmArmController.goal_response_callback(controller, future, "gripper", "cup", OPEN_GRIPPER)
    assert any("Command rejected: name=cup, controller=gripper" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
@pytest.mark.parametrize("controller_name, result, expected", [("arm", ArmResult(error_code=-1), "success=False"), ("gripper", GripperResult(position=0.0, reached_goal=False, stalled=True), "success=False")])
def test_unsuccessful_live_result_is_reported(controller_name, result, expected):
    controller = node()
    future = ActionFuture(ResultResponse(result=result))
    ComputRhythmArmController.get_result_callback(controller, future, controller_name, "home", [0.0, -1.0, 0.0, 1.0] if controller_name == "arm" else OPEN_GRIPPER)
    assert any(expected in message for message in controller.logger.messages)
