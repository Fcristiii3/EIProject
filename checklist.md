## checklist

- [ ] Basic arm movement
- [] Camera recognition of the cup placement/position on the table
- [ ]initialization of the arm movement range to have a known coordinate system
- [] Variable tempo of the hand
- [] Translation from camera coordinates to arm movement
- [] Grabbing the cup, randomness of choices (tap or grab), and waiting for player action

---------------------------------------------------

- Use manipulatorx software to connect physical robot with our ros2 environment
- Calibrate camera with the sticker at the base of the robot- object placement, hand position
- Game loop (when is the humans turn or robot's turn)
- Tempo variable 
- Tempo and game loop: Topic for tempo, in order to compute the cooldown of the robot until its next turn
-1000000 bps protocol 2

joint_state_broadcaster joint_state_broadcaster/JointStateBroadcaster          active
gripper_controller      position_controllers/GripperActionController           active
arm_controller          joint_trajectory_controller/JointTrajectoryController  active


Published robot coordinates: x=316.0211037900929, y=-519.6713088260402 (camera node)

[INFO] [1795032502.878126288] [comput_rhythm_arm_controller]: Live action feedback received -> control_msgs.action.FollowJointTrajectory_Feedback(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1795032502, nanosec=240275554), frame_id=''), joint_names=['joint1', 'joint2', 'joint3', 'joint4'], desired=trajectory_msgs.msg.JointTrajectoryPoint(positions=[324.5622925325557, -881.1795683317769, 0.2, 0.3], velocities=[-18.842884086177037, 796.4274052627574, 0.0, 0.0], accelerations=[0.0, 0.0, 0.0, 0.0], effort=[], time_from_start=builtin_interfaces.msg.Duration(sec=0, nanosec=546715423)), actual=trajectory_msgs.msg.JointTrajectoryPoint(positions=[-2.0694557179012918e-13, -1.000155473701645, 0.1994175024249265, 0.299126253637493], velocities=[0.0, 0.0, 0.0, 0.0], accelerations=[], effort=[], time_from_start=builtin_interfaces.msg.Duration(sec=0, nanosec=546715423)), error=trajectory_msgs.msg.JointTrajectoryPoint(positions=[324.56229253255594, -880.1794128580752, 0.000582497575073504, 0.0008737463625069775], velocities=[0.0, 0.0, 0.0, 0.0], accelerations=[], effort=[], time_from_start=builtin_interfaces.msg.Duration(sec=0, nanosec=0)), multi_dof_joint_names=[], multi_dof_desired=trajectory_msgs.msg.MultiDOFJointTrajectoryPoint(transforms=[], velocities=[], accelerations=[], time_from_start=builtin_interfaces.msg.Duration(sec=0, nanosec=0)), multi_dof_actual=trajectory_msgs.msg.MultiDOFJointTrajectoryPoint(transforms=[], velocities=[], accelerations=[], time_from_start=builtin_interfaces.msg.Duration(sec=0, nanosec=0)), multi_dof_error=trajectory_msgs.msg.MultiDOFJointTrajectoryPoint(transforms=[], velocities=[], accelerations=[], time_from_start=builtin_interfaces.msg.Duration(sec=0, nanosec=0)))
[INFO] [1795032502.954740004] [comput_rhythm_arm_controller]: Command result: name=camera, controller=arm, target=[316.0211037900929, -520.1713088260402, 0.2, 0.3], achieved_error_code=0, success=True.