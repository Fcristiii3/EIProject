DEFAULT_TIMEOUT = 30.0


def pytest_addoption(parser):
    parser.addoption('--robot-arm-mode', choices=['mock', 'live'], default='mock', help='Controller mode for the integration test.')
    parser.addoption('--robot-arm-timeout', type=float, default=DEFAULT_TIMEOUT, help=f'Timeout in seconds for node startup and each command, default: {DEFAULT_TIMEOUT}.')
