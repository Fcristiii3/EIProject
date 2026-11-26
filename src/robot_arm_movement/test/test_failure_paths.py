"""Deterministic fault-injection tests for action rejection, failures, and step aborts."""

from collections import namedtuple
import pytest
from robot_arm_movement.arm_controller_node import OPEN_GRIPPER, ComputRhythmArmController


class FailureLogger:
    """In-memory logger capturing info, warn, and error messages."""

    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warn(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


GoalHandle = namedtuple("GoalHandle", ["accepted"])
ResultResponse = namedtuple("ResultResponse", ["result"])
ArmResult = namedtuple("ArmResult", ["error_code"])
GripperResult = namedtuple("GripperResult", ["position", "reached_goal", "stalled"])


class ActionFuture:
    """Future stand-in whose result() returns the response held by the future."""

    __slots__ = ("_response",)

    def __init__(self, response=None):
        self._response = response

    def done(self):
        return self._response is not None

    def result(self):
        return self._response


class Controller:
    """Slotted controller fake for failure-path unit tests."""

    __slots__ = ("_initialization_goals_pending", "logger", "_active_step_pending", "_active_step_failed", "_pending_steps", "_sequence_active", "_command_queue", "statuses", "init_done_calls", "next_step_calls")

    def __init__(self):
        self.logger = FailureLogger()
        self._initialization_goals_pending, self._active_step_pending, self._active_step_failed = 2, 2, False
        self._pending_steps, self._sequence_active, self._command_queue, self.statuses = ["remaining_step"], True, [], []
        self.init_done_calls, self.next_step_calls = 0, 0

    def get_logger(self):
        return self.logger

    def _publish_status(self, status):
        self.statuses.append(status)

    def _drain_command_queue(self):
        pass

    def initialization_done_callback(self, future):
        self.init_done_calls += 1

    def _start_next_step(self):
        self.next_step_calls += 1


@pytest.mark.fault_injection
def test_rejected_goal_is_logged():
    controller = Controller()
    future = ActionFuture(GoalHandle(accepted=False))
    ComputRhythmArmController.goal_response_callback(controller, future, "arm", "home", [0.0, -1.0, 0.0, 1.0])
    assert any("Command rejected: name=home, controller=arm" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_arm_failure_does_not_complete_initialization():
    controller = Controller()
    result, callback_called = ArmResult(error_code=-1), []
    ComputRhythmArmController.get_result_callback(controller, ActionFuture(ResultResponse(result=result)), "arm", "home", [0.0, -1.0, 0.0, 1.0], callback_called.append)
    assert controller._initialization_goals_pending == 2
    assert callback_called == [result]
    assert any("success=False" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_gripper_failure_does_not_complete_initialization():
    controller = Controller()
    result, callback_called = GripperResult(position=0.0, reached_goal=False, stalled=True), []
    ComputRhythmArmController.get_result_callback(controller, ActionFuture(ResultResponse(result=result)), "gripper", "home", OPEN_GRIPPER, callback_called.append)
    assert controller._initialization_goals_pending == 2
    assert callback_called == [result]
    assert any("success=False" in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_failed_initialization_result_still_completes_without_lockout():
    controller = Controller()
    ComputRhythmArmController._initialization_goal_done_callback(controller, ArmResult(error_code=-1))
    assert controller._initialization_goals_pending == 1
    assert controller.init_done_calls == 0
    ComputRhythmArmController._initialization_goal_done_callback(controller, GripperResult(position=0.0, reached_goal=False, stalled=True))
    assert controller._initialization_goals_pending == 0
    assert controller.init_done_calls == 1
    assert any("Initialization arm goal reported error" in m for m in controller.logger.messages)
    assert any("Initialization gripper stalled/incomplete" in m for m in controller.logger.messages)


@pytest.mark.fault_injection
def test_stalled_step_gripper_warns_and_continues_sequence():
    controller = Controller()
    ComputRhythmArmController._on_step_controller_done(controller, ArmResult(error_code=0))
    ComputRhythmArmController._on_step_controller_done(controller, GripperResult(position=-0.032, reached_goal=False, stalled=True))
    assert controller._active_step_failed is False
    assert controller._sequence_active is True
    assert controller._pending_steps == ["remaining_step"]
    assert controller.statuses == []
    assert controller.next_step_calls == 1
    assert any("Step gripper stalled/incomplete" in m for m in controller.logger.messages)


@pytest.mark.fault_injection
def test_failed_step_aborts_sequence_and_publishes_error():
    controller = Controller()
    ComputRhythmArmController._on_step_controller_done(controller, ArmResult(error_code=-1))
    ComputRhythmArmController._on_step_controller_done(controller, GripperResult(position=0.0, reached_goal=True, stalled=False))
    assert controller._sequence_active is False
    assert controller._pending_steps == []
    assert controller.statuses == ["error"]
