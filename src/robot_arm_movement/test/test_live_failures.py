from collections import namedtuple

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


class Client:

    __slots__ = ("available", "service")

    def __init__(self, available=True, service=True):
        self.available = available
        self.service = service

    def wait_for_server(self, timeout_sec):
        return self.available

    def wait_for_service(self, timeout_sec):
        return self.service


class ActionFuture:
    """Future stand-in whose result() returns the response held by the future."""

    __slots__ = ("_response",)

    def __init__(self, response=None):
        self._response = response

    def done(self):
        return self._response is not None

    def result(self):
        return self._response


class ActionClient:
    """Action client fake dispatching call_async onto a configured future."""

    __slots__ = ("future",)

    def __init__(self, future=None):
        self.future = future

    def wait_for_service(self, timeout_sec):
        return True

    def call_async(self, request):
        return self.future


ParameterValue = namedtuple("ParameterValue", ["bool_value", "double_value"])


class LiveParameter:
    """Parameter value holder for the runtime-discovery test doubles."""

    __slots__ = ()

    def get_parameter_value(self):
        return ParameterValue(bool_value=False, double_value=15.0)


def live_parameters(_name=None):
    return LiveParameter()


GoalHandle = namedtuple("GoalHandle", ["accepted"])
ResultResponse = namedtuple("ResultResponse", ["result"])
ArmResult = namedtuple("ArmResult", ["error_code"])
GripperResult = namedtuple("GripperResult", ["position", "reached_goal", "stalled"])
StringArrayValue = namedtuple("StringArrayValue", ["string_array_value"])
StringValue = namedtuple("StringValue", ["string_value"])
DiscoveredResponse = namedtuple("DiscoveredResponse", ["values"])


def default_create_client(service, name):
    return Client()


class Controller:
    """Slotted controller fake backing the runtime-discovery tests."""

    __slots__ = (
        "_arm_action_client",
        "_gripper_action_client",
        "create_client",
        "get_parameter",
        "logger",
        "pose_split_arm_gripper",
        "runtime_ready",
        "arm_joint_names",
        "gripper_joint_name",
    )

    def __init__(self):
        self.logger = Logger()
        self.get_parameter = live_parameters
        self.create_client = default_create_client
        self.pose_split_arm_gripper = 4
        self.runtime_ready = False
        self.arm_joint_names = None
        self.gripper_joint_name = None
        self._arm_action_client = Client()
        self._gripper_action_client = Client()

    def get_logger(self):
        return self.logger


def node(**values):
    controller = Controller()
    for key, value in values.items():
        setattr(controller, key, value)
    return controller


@pytest.mark.fault_injection
def test_missing_action_server_does_not_enable_runtime(monkeypatch):
    controller = node(
        get_parameter=live_parameters,
        _arm_action_client=Client(available=False),
        _gripper_action_client=Client(),
    )

    ComputRhythmArmController._discover_runtime(controller)

    assert controller.runtime_ready is False
    assert any("Action server availability" in message for message in controller.logger.messages)


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
    assert any("parameter services are unavailable" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_parameter_timeout_does_not_enable_runtime(monkeypatch):
    future = ActionFuture()
    controller = node(
        get_parameter=live_parameters,
        _arm_action_client=Client(),
        _gripper_action_client=Client(),
        create_client=lambda service, name: ActionClient(future=future),
    )
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete", lambda *args, **kwargs: None)

    ComputRhythmArmController._discover_runtime(controller)

    assert controller.runtime_ready is False
    assert any("discovery timed out" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
@pytest.mark.parametrize("arm_joints, gripper_joint", [([], "gripper_left_joint"), (["joint1", "joint2", "joint3"], "gripper_left_joint"), (["joint1", "joint2", "joint3", "joint4"], "")])
def test_invalid_discovered_joints_do_not_enable_runtime(monkeypatch, arm_joints, gripper_joint):
    arm_future = ActionFuture(DiscoveredResponse(values=[StringArrayValue(string_array_value=arm_joints)]))
    gripper_future = ActionFuture(DiscoveredResponse(values=[StringValue(string_value=gripper_joint)]))
    controller = node(
        get_parameter=live_parameters,
        _arm_action_client=Client(),
        _gripper_action_client=Client(),
        create_client=lambda service, name: ActionClient(future=gripper_future if "gripper" in name else arm_future),
    )
    monkeypatch.setattr("robot_arm_movement.arm_controller_node.rclpy.spin_until_future_complete", lambda *args, **kwargs: None)

    ComputRhythmArmController._discover_runtime(controller)

    assert controller.runtime_ready is False
    assert any("Invalid discovered joints" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_rejected_live_goal_is_reported():
    controller = node()
    future = ActionFuture(GoalHandle(accepted=False))

    ComputRhythmArmController.goal_response_callback(controller, future, "gripper", "cup", 0.015)

    assert any("Command rejected: name=cup, controller=gripper" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
@pytest.mark.parametrize(
    "controller_name, result, expected",
    [
        ("arm", ArmResult(error_code=-1), "success=False"),
        ("gripper", GripperResult(position=0.0, reached_goal=False, stalled=True), "success=False"),
    ],
)
def test_unsuccessful_live_result_is_reported(controller_name, result, expected):
    controller = node()
    future = ActionFuture(ResultResponse(result=result))

    ComputRhythmArmController.get_result_callback(controller, future, controller_name, "home", [0.0, -1.0, 0.0, 1.0] if controller_name == "arm" else 0.015)

    assert any(expected in message for message in controller.logger.messages)
