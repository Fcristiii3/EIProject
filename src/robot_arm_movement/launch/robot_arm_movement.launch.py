"""Unified launch file for robot_arm_movement supporting mock, fake, gazebo, hardware, and attach modes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    port_name = LaunchConfiguration("port_name").perform(context)
    start_rviz = LaunchConfiguration("start_rviz").perform(context)
    auto_return_home = LaunchConfiguration("auto_return_home").perform(context).strip().lower() == "true"
    grab_hold_duration_sec = float(LaunchConfiguration("grab_hold_duration_sec").perform(context))
    tempo_scale = float(LaunchConfiguration("tempo_scale").perform(context))
    discovery_timeout = float(LaunchConfiguration("controller_discovery_timeout_sec").perform(context))
    camera_action_mode = LaunchConfiguration("camera_action_mode").perform(context)
    gripper_closed_position = float(LaunchConfiguration("gripper_closed_position").perform(context))
    gripper_open_position = float(LaunchConfiguration("gripper_open_position").perform(context))
    if mode not in ("hardware", "fake", "gazebo", "attach", "mock"): raise ValueError(f"Unsupported launch mode: {mode!r}")
    actions = []
    if mode == "hardware":
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource([PathJoinSubstitution([FindPackageShare("open_manipulator_x_bringup"), "launch", "hardware.launch.py"])]), launch_arguments={"port_name": port_name, "start_rviz": start_rviz, "use_sim": "false", "use_fake_hardware": "false"}.items()))
    elif mode == "gazebo":
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource([PathJoinSubstitution([FindPackageShare("open_manipulator_x_bringup"), "launch", "gazebo.launch.py"])]), launch_arguments={"start_rviz": start_rviz, "use_sim": "true"}.items()))
    elif mode == "fake":
        urdf_xml = ParameterValue(Command([PathJoinSubstitution([FindExecutable(name="xacro")]), " ", PathJoinSubstitution([FindPackageShare("open_manipulator_x_description"), "urdf", "open_manipulator_x_robot.urdf.xacro"]), " use_sim:=false use_fake_hardware:=true fake_sensor_commands:=true port_name:=", port_name]), value_type=str)
        controllers_yaml = PathJoinSubstitution([FindPackageShare("robot_arm_movement"), "config", "controllers.yaml"])
        jsb_spawner = Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"], output="screen")
        arm_spawner = Node(package="controller_manager", executable="spawner", arguments=["arm_controller", "--controller-manager", "/controller_manager"], output="screen")
        grip_spawner = Node(package="controller_manager", executable="spawner", arguments=["gripper_controller", "--controller-manager", "/controller_manager"], output="screen")
        actions.extend([
            Node(package="robot_state_publisher", executable="robot_state_publisher", parameters=[{"robot_description": urdf_xml, "use_sim_time": False}], output="screen"),
            Node(package="controller_manager", executable="ros2_control_node", parameters=[{"robot_description": urdf_xml}, controllers_yaml], output="both"),
            jsb_spawner,
            RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[arm_spawner, grip_spawner])),
        ])
    arm_params = {"mock_mode_enabled": mode == "mock", "auto_return_home": auto_return_home, "grab_hold_duration_sec": grab_hold_duration_sec, "tempo_scale": tempo_scale, "controller_discovery_timeout_sec": discovery_timeout, "camera_action_mode": camera_action_mode, "gripper_closed_position": gripper_closed_position, "gripper_open_position": gripper_open_position, "use_sim_time": mode == "gazebo"}
    actions.append(Node(package="robot_arm_movement", executable="arm_controller_node", name="comput_rhythm_arm_controller", parameters=[arm_params], output="screen"))
    actions.append(Node(package="camera", executable="camera_node", output="screen", condition=IfCondition(LaunchConfiguration("launch_camera"))))
    return actions


def generate_launch_description():
    """Generate launch description for robot_arm_movement."""
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="mock", description="Bringup mode: hardware, fake, gazebo, attach, or mock."),
        DeclareLaunchArgument("port_name", default_value="/dev/ttyUSB0", description="Serial device for Dynamixel hardware."),
        DeclareLaunchArgument("launch_camera", default_value="false", description="Launch EIProject camera_node alongside arm controller."),
        DeclareLaunchArgument("auto_return_home", default_value="true", description="Enable multi-step tap/grab/place sequences."),
        DeclareLaunchArgument("grab_hold_duration_sec", default_value="0.0", description="Optional auto put-back timer after grab (0.0 waits for place/reset)."),
        DeclareLaunchArgument("tempo_scale", default_value="1.0", description="Initial trajectory tempo multiplier."),
        DeclareLaunchArgument("controller_discovery_timeout_sec", default_value="15.0", description="Timeout for URDF and controller discovery."),
        DeclareLaunchArgument("start_rviz", default_value="false", description="Start RViz2 when supported by bringup mode."),
        DeclareLaunchArgument("camera_action_mode", default_value="random", description="Camera action selection: random, tap, or grab."),
        DeclareLaunchArgument("gripper_closed_position", default_value="-0.027", description="Gripper closed position target."),
        DeclareLaunchArgument("gripper_open_position", default_value="-0.045", description="Gripper open position target."),
        OpaqueFunction(function=_launch_setup),
    ])
