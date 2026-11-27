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


---------------------------------------------------------
camera fixes needed (`src/camera/camera/camera.py`):

- [ ] Subscribe to `/comput_game/arm_status` (`std_msgs/msg/String`) instead of ending `RobotExecuting` on a fixed `self.tempo` timer (`1.0–3.0s`):
  - A 4-step `grab` or `place` sequence takes ~`3.1 * tempo_scale` seconds, so the timer can switch back to `"Human"` while the arm is still moving.
  - Transition `RobotExecuting` -> `"Human"` only after receiving `"done:tap"`, `"done:grab"`, `"done:place"`, `"ready"`, or `"error"` (with an optional safety timeout).
- [ ] Explicitly sync cup-held state and publish `"place"` to `/comput_game/target_action` (`std_msgs/msg/String`) when the robot is holding the cup:
  - Currently `camera.py` only publishes `/comput_game/camera_coordinates` (using `cupHistory` with `z=0.1` when `cupID=1` is not visible).
  - Track `cup_held` from `/comput_game/arm_status` (`"done:grab"` -> `True`, `"done:place"`/`"done:reset"`/`"done:home"` -> `False`) and publish `"place"` on `/comput_game/target_action` on the robot's next turn (arm controller now also safeguards `holding_cup=True` on incoming coordinates by placing the cup back at its remembered table pose).
- [ ] Guard `cupHistory` and `lastCupPixelPos` while the cup is held/absent:
  - Freeze or clear `cupHistory` while `cup_held` is `True` (until `"done:place"` and fresh `cupID=1` detection), so stale history does not cause spurious `FIST` / `isHandNearCup` triggers.
- [ ] Publish tempo changes to `/comput_game/tempo` (`std_msgs/msg/Float64`):
  - Currently `self.tempo = random.uniform(1.0, 3.0)` is only used locally in `camera.py`, so `arm_controller_node` stays on its default `tempo_scale` (`1.0`).
