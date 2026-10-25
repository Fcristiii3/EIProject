from types import SimpleNamespace

import pytest

from robot_arm_movement.arm_controller_node import ComputRhythmArmController


class FailureLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(message)

    def warn(self, message):
        self.messages.append(message)

    def error(self, message):
        self.messages.append(message)


def controller_for_failure_test():
    logger = FailureLogger()
    return SimpleNamespace(
        get_logger=lambda: logger,
        _initialization_goals_pending=2,
        logger=logger,
    )


@pytest.mark.fault_injection
def test_rejected_goal_is_logged():
    controller = controller_for_failure_test()
    goal = SimpleNamespace(accepted=False)
    future = SimpleNamespace(result=lambda: goal)

    ComputRhythmArmController.goal_response_callback(controller, future, 'arm', 'home', [0.0, -1.0, 0.0, 1.0])

    assert any('Command rejected: name=home, controller=arm' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_arm_failure_does_not_complete_initialization():
    controller = controller_for_failure_test()
    result = SimpleNamespace(error_code=-1)
    callback_called = []

    ComputRhythmArmController.get_result_callback(controller, SimpleNamespace(result=result), 'arm', 'home', [0.0, -1.0, 0.0, 1.0], callback_called.append)

    assert controller._initialization_goals_pending == 2
    assert callback_called == [result]
    assert any('success=False' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_gripper_failure_does_not_complete_initialization():
    controller = controller_for_failure_test()
    result = SimpleNamespace(position=0.0, reached_goal=False, stalled=True)
    callback_called = []

    ComputRhythmArmController.get_result_callback(controller, SimpleNamespace(result=result), 'gripper', 'home', 0.015, callback_called.append)

    assert controller._initialization_goals_pending == 2
    assert callback_called == [result]
    assert any('success=False' in message for message in controller.logger.messages)


@pytest.mark.fault_injection
def test_failed_initialization_result_is_not_counted():
    controller = controller_for_failure_test()
    result = SimpleNamespace(error_code=-1)

    ComputRhythmArmController._initialization_goal_done_callback(controller, result)

    assert controller._initialization_goals_pending == 2
