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




----------------------------------------------------
freshest issues:


joints 1-4 work fine (at least on the setup). gripper is not responding for some reason

gripper works in the 300-360 degrees range. (from dynamixel app, when controlled manually).

camera has bugs when we are usign the history, without the cup beign there. especially the fist near the cup

INFO] [1795608858.385671679] [camera_node]: Cup stable at: x=0.422, y=0.068

[INFO] [1795608858.428815335] [camera_node]: Publishing 0.4215482908410768 and 0.06750054533770972


---------------------------------------------------------


with the current implementation, the gripper works between 0.05 (fully closed) - 0.03 (it is open enough for any possible interaction)
///// another try, and the values are -0.027(closed) and -0.045(open)
#template for how i moved it: 
ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.03, max_effort: 5.0}}"



auto_return_home and a ready signal after the intiialization, for some reason the robot was stuck and not ready to accept any point coordinate and execute it.

after any movement, the robot hand stays there. that's an issue.
left and right are inverted (i changed camera camera_scale_xyz, y to -1.0)

//what to change: gripper functionality
game logic ( claw open, touch, or grab)
image distoriton; flatten the image, maybe use some reference point,stickers, whatever.


