"""Pytest CLI configuration for robot_arm_movement tests."""

DEFAULT_TIMEOUT = 30.0


def pytest_addoption(parser):
    """Add --robot-arm-mode and --robot-arm-timeout pytest options."""
    parser.addoption("--robot-arm-mode", choices=["mock", "fake", "live"], default="mock", help="Controller mode for the integration test.")
    parser.addoption("--robot-arm-timeout", type=float, default=DEFAULT_TIMEOUT, help=f"Timeout in seconds for node startup and each command, default: {DEFAULT_TIMEOUT}.")
