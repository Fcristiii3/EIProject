""" purpose: this node controls robot arm movement. """
#imports
import functools, random, sys, rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Point
from rclpy.action import ActionClient
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint
#node class
class ComputRhythmArmController(Node):
    def __init__(self):
        super().__init__('comput_rhythm_arm_controller')
        self._arm_action_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self._gripper_action_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.target_sub = self.create_subscription(String, '/comput_game/target_action', self.game_action_callback, 10)
        self.coord_sub = self.create_subscription(Point, '/comput_game/camera_coordinates', self.camera_coord_callback, 10)
        self.declare_parameter('tempo_scale', 1.0)
        self.declare_parameter('mock_mode_enabled', True)
        self.declare_parameter('mock_execution_delay', 0.3)
        self.declare_parameter('controller_discovery_timeout_sec', 15.0)
        self.arm_joint_names=None
        self.gripper_joint_name=None
        self.runtime_ready=False
        self.home_pose = [0.0, -1.0, 0.0, 1.0, 0.015]
        self.tap_pose = [0.1, -0.4, 0.2, 0.3, 0.015]
        self.grab_pose = [0.1, -0.4, 0.2, 0.3, -0.015]
        self.pose_split_arm_gripper=4
        self.available_choices = ['tap', 'grab']
        self.interactive_commands = ['interact', 'cup', 'box']
        self.reset_commands = ['cloth', 'reset', 'home']
        self._initialization_goals_pending = 2
        self._discover_runtime()
        self.get_logger().info('ComputRhythmArmController initialized in Dual-Mode. Executing initial calibration / home setup.')
        self.send_trajectory_goal(self.home_pose, 2.0, command_name='home', callback=self._initialization_goal_done_callback)
    def initialization_done_callback(self, future):
        self.get_logger().info('Arm initialization range established successfully. Ready for rhythm game commands.')
    def _initialization_goal_done_callback(self, future):
        if future is not None and hasattr(future, 'error_code') and future.error_code != 0:
            self.get_logger().error(f'Initialization arm goal failed: error_code={future.error_code}.')
            return
        if future is not None and hasattr(future, 'reached_goal') and (not future.reached_goal or future.stalled):
            self.get_logger().error(f'Initialization gripper goal failed: reached_goal={future.reached_goal}, stalled={future.stalled}.')
            return
        self._initialization_goals_pending -= 1
        if self._initialization_goals_pending == 0:
            self.initialization_done_callback(future)
    def _discover_runtime(self):
        if self.get_parameter('mock_mode_enabled').get_parameter_value().bool_value:
            self.get_logger().info('Mock mode enabled; skipping live controller discovery.')
            self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
            self.gripper_joint_name = 'gripper_left_joint'
            self.runtime_ready = True
            return
        self.get_logger().info('Waiting for arm and gripper controllers to be ready...')
        timeout_sec = self.get_parameter('controller_discovery_timeout_sec').get_parameter_value().double_value
        arm_ready = self._arm_action_client.wait_for_server(timeout_sec=timeout_sec)
        gripper_ready = self._gripper_action_client.wait_for_server(timeout_sec=timeout_sec)
        if not arm_ready or not gripper_ready:
            self.get_logger().error(f'Action server availability: arm={arm_ready}, gripper={gripper_ready}.')
            return
        arm_client = self.create_client(GetParameters, '/arm_controller/get_parameters')
        gripper_client = self.create_client(GetParameters, '/gripper_controller/get_parameters')
        if not arm_client.wait_for_service(timeout_sec=timeout_sec) or not gripper_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error('Controller parameter services are unavailable.')
            return
        arm_request = GetParameters.Request(names=['joints'])
        gripper_request = GetParameters.Request(names=['joint'])
        arm_future = arm_client.call_async(arm_request)
        gripper_future = gripper_client.call_async(gripper_request)
        rclpy.spin_until_future_complete(self, arm_future, timeout_sec=timeout_sec)
        rclpy.spin_until_future_complete(self, gripper_future, timeout_sec=timeout_sec)
        if not arm_future.done() or not gripper_future.done():
            self.get_logger().error('Controller parameter discovery timed out.')
            return
        self.arm_joint_names = list(arm_future.result().values[0].string_array_value)
        self.gripper_joint_name = gripper_future.result().values[0].string_value
        if len(self.arm_joint_names) != self.pose_split_arm_gripper or not self.gripper_joint_name:
            self.get_logger().error(f'Invalid discovered joints: arm={self.arm_joint_names}, gripper={self.gripper_joint_name}.')
            return
        self.runtime_ready = True
    def _resolve_target_pose(self, action_type, tempo_scale, msg_coords=None):
        chosen_action = random.choice(self.available_choices)
        self.get_logger().info(f'Randomized behavior choice -> action: "{chosen_action}", mode: "{action_type}", tempo_scale: {tempo_scale}')
        if action_type == 'camera':
            z_offset = 0.015 if chosen_action == 'tap' else -0.015
            return [msg_coords.x, msg_coords.y - 0.5, msg_coords.z + 0.2, 0.3, z_offset]
        if action_type == 'interactive':
            return self.tap_pose if chosen_action == 'tap' else self.grab_pose
        if action_type == 'reset':
            return self.home_pose
        return self.home_pose
    def camera_coord_callback(self, msg):
        self.get_logger().info(f'Camera coordinates received -> x: {msg.x:.3f}, y: {msg.y:.3f}, z: {msg.z:.3f}')
        tempo_scale = self.get_parameter('tempo_scale').get_parameter_value().double_value
        target_pose = self._resolve_target_pose('camera', tempo_scale, msg_coords=msg)
        duration_sec = max(0.15, 1.0 * tempo_scale)
        self.send_trajectory_goal(target_pose, duration_sec, command_name='camera')
    def game_action_callback(self, msg):
        command = msg.data.strip().lower()
        self.get_logger().info(f'Game state command received -> command: "{command}"')
        if command not in self.interactive_commands and command not in self.reset_commands:
            self.get_logger().warn(f'Invalid game command received -> command: "{command}". Aborting execution.')
            return
        tempo_scale = self.get_parameter('tempo_scale').get_parameter_value().double_value
        if command in self.interactive_commands:
            target_pose = self._resolve_target_pose('interactive', tempo_scale)
        else:
            target_pose = self._resolve_target_pose('reset', tempo_scale)
        duration_sec = max(0.15, 1.0 * tempo_scale)
        self.send_trajectory_goal(target_pose, duration_sec, command_name=command)
    def _complete_mock_goal(self, callback):
        self.get_logger().info(f'Mock {self._mock_controller} goal completed: command={self._mock_command}, target={self._mock_target}.')
        if callback:
            callback(None)
    def _complete_mock_execution(self):
        self._mock_timer.cancel()
        self._mock_controller = 'arm'
        self._complete_mock_goal(self._mock_callback)
        self._mock_controller = 'gripper'
        self._mock_target = self._mock_gripper_target
        self._complete_mock_goal(self._mock_callback)
    def send_trajectory_goal(self, positions, duration_sec, command_name='command', callback=None,max_effort=5.0):
        arm_positions = positions[:self.pose_split_arm_gripper]
        gripper_position = positions[self.pose_split_arm_gripper]
        mock_enabled = self.get_parameter('mock_mode_enabled').get_parameter_value().bool_value
        if mock_enabled:
            self._mock_command = command_name
            self._mock_target = arm_positions
            self._mock_gripper_target = gripper_position
            self._mock_controller = 'arm'
            self._mock_callback = callback
            self._mock_timer = self.create_timer(self.get_parameter('mock_execution_delay').get_parameter_value().double_value, self._complete_mock_execution)
            return
        if not self.runtime_ready and not mock_enabled:
            self.get_logger().error('Action server offline and mock_mode_enabled is False! Dropping goal dispatch.')
            return
        self.get_logger().info(f'Command sent: name={command_name}, arm_target={arm_positions}, gripper_target={gripper_position}.')
        arm_goal = FollowJointTrajectory.Goal()
        arm_goal.trajectory.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = arm_positions
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        arm_point.time_from_start = Duration(sec=sec,nanosec=nanosec)
        arm_goal.trajectory.points.append(arm_point)
        self.get_logger().info(f'Gripper control Action server online. Dispatching live trajectory goal -> target_positions: {gripper_position}, max_effort: {max_effort:.2f}N.')
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position = float(gripper_position)
        gripper_goal.command.max_effort = max_effort
        arm_goal_future = self._arm_action_client.send_goal_async(arm_goal, feedback_callback=self.feedback_callback)
        arm_goal_future.add_done_callback(functools.partial(self.goal_response_callback, controller='arm', command_name=command_name, target=arm_positions, callback=callback))
        gripper_goal_future = self._gripper_action_client.send_goal_async(gripper_goal, feedback_callback=self.feedback_callback)
        gripper_goal_future.add_done_callback(functools.partial(self.goal_response_callback, controller='gripper', command_name=command_name, target=gripper_position, callback=callback))
    def goal_response_callback(self, future, controller, command_name, target, callback=None):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'Command rejected: name={command_name}, controller={controller}, target={target}.')
            return
        self.get_logger().info(f'Command accepted: name={command_name}, controller={controller}, target={target}.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(functools.partial(self.get_result_callback, controller=controller, command_name=command_name, target=target, callback=callback))
    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(f'Live action feedback received -> {fb}')
    def get_result_callback(self, future, controller, command_name, target, callback=None):
        result = future.result().result
        if controller == 'arm':
            success = result.error_code == 0
            self.get_logger().info(f'Command result: name={command_name}, controller=arm, target={target}, achieved_error_code={result.error_code}, success={success}.')
        else:
            success = result.reached_goal and not result.stalled
            self.get_logger().info(f'Command result: name={command_name}, controller=gripper, target={target}, achieved_position={result.position}, success={success}.')
        if callback:
            callback(result)
#main function
def main(args=None):
    rclpy.init(args=args)
    node = ComputRhythmArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt detected. Shutting down node cleanly...')
    except Exception as e:
        node.get_logger().error(f'An exception occurred during node spin -> exception: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()
# main function conditional execution
if __name__ == '__main__':
    main()
