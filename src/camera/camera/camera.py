import random
import time
import cv2
import mediapipe as mp
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.bridge = CvBridge()

        # Publishers
        self.robot_pub = self.create_publisher(Point, "/comput_game/camera_coordinates", 10)
        self.action_pub = self.create_publisher(String, "/comput_game/target_action", 10)
        self.tempo_pub = self.create_publisher(Float64, "/comput_game/tempo", 10)

        # Subscriptions
        self.arm_status_sub = self.create_subscription(String, "/comput_game/arm_status", self.arm_status_callback, 10)

        # Game state tracking
        self.tempo = random.uniform(1.0, 3.0)
        self.cooldownActive = False
        self.cooldownStartTime = 0.0
        self.cup_held = False
        self.arm_executing = False

        # --- 4-CORNER HOMOGRAPHY ARENA CONFIGURATION ---
        self.cupID = 1

        self.ARENA_MARKER_WORLD_POS = {
            5: [0.28, 0.30],
            2: [0.52, 0.30],
            3: [0.52, -0.30],
            4: [0.28, -0.30],
        }

        self.homography_matrix = None

        # ArUco marker configuration
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        # MediaPipe Hand Landmarker
        baseOptions = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=baseOptions,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cupHistory = []
        self.historyLength = 5
        self.cupThreshold = 0.015
        self.lastCupPixelPos = None
        self.OFFSET_X = 0.014  
        self.OFFSET_Y = 0.03 

        self.gameState = "Human"

        # Camera capture setup
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow("camera")
        self.timer = self.create_timer(0.05, self.timer_callback)

        self._publish_tempo()

    def _publish_tempo(self):
        msg = Float64()
        msg.data = float(self.tempo)
        self.tempo_pub.publish(msg)
        self.get_logger().info(f"Published tempo scale -> {self.tempo:.2f}")

    def update_homography(self, marker_centers_map):
        """Compute homography matrix H matching exact Marker IDs to physical world points."""
        required_ids = list(self.ARENA_MARKER_WORLD_POS.keys())

        if all(m_id in marker_centers_map for m_id in required_ids):
            img_pts = np.array([marker_centers_map[m_id] for m_id in required_ids], dtype=np.float32)
            world_pts = np.array([self.ARENA_MARKER_WORLD_POS[m_id] for m_id in required_ids], dtype=np.float32)

            H, status = cv2.findHomography(img_pts, world_pts)
            if H is not None:
                self.homography_matrix = H
                return True
        return False

    def pixel_to_world(self, pixel_pt):
        """Unwarp pixel coordinate (u, v) directly into robot base (x, y) meters."""
        if self.homography_matrix is None:
            return None
        src = np.array([[[pixel_pt[0], pixel_pt[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.homography_matrix)
        world_x = float(dst[0][0][0]) + self.OFFSET_X
        world_y = float(dst[0][0][1]) + self.OFSSET_Y
        return world_x, world_y

    def arm_status_callback(self, msg: String):
        status = msg.data.strip()
        self.get_logger().info(f"Received arm status -> {status}")

        if status == "done:grab":
            self.cup_held = True
            self.cupHistory.clear()
            self.lastCupPixelPos = None
            self.get_logger().info("Cup registered as HELD. Guarded cup history.")
        elif status in ("done:place", "done:reset", "done:home"):
            self.cup_held = False
            self.get_logger().info("Cup registered as RELEASED/PLACED.")

        if self.gameState == "RobotExecuting" and status in ("done:tap", "done:grab", "done:place", "ready", "error"):
            self.get_logger().info(f"Arm finished motion with status '{status}'. Transitioning to Human turn!")
            self.arm_executing = False
            self.cooldownActive = False
            self.gameState = "Human"

    def timer_callback(self):
        for _ in range(2):
            self.cap.grab()

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("No frames captured from camera.")
            return

        corners, ids, _ = self.detector.detectMarkers(frame)
        marker_corners_map = {}
        marker_centers_map = {}

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for idx, marker_id in enumerate(ids.flatten()):
                marker_id_int = int(marker_id)
                pts = corners[idx][0]
                center = np.mean(pts, axis=0)
                marker_corners_map[marker_id_int] = corners[idx]
                marker_centers_map[marker_id_int] = center

        # Update homography matrix whenever all 4 arena markers (5, 2, 3, 4) are visible
        self.update_homography(marker_centers_map)

        # Track cup position using homography mapping
        if not self.cup_held:
            if self.cupID in marker_centers_map and self.homography_matrix is not None:
                cup_pixel = marker_centers_map[self.cupID]
                self.lastCupPixelPos = cup_pixel
                world_pos = self.pixel_to_world(cup_pixel)

                if world_pos is not None:
                    rel_x, rel_y = world_pos
                    self.cupHistory.append((rel_x, rel_y))
                    if len(self.cupHistory) > self.historyLength:
                        self.cupHistory.pop(0)

        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
        detectionResult = self.landmarker.detect(mpImage)

        gesture = None
        currentHandLandmarks = None

        if detectionResult.hand_landmarks:
            for handLandmarks in detectionResult.hand_landmarks:
                self.drawLandmarks(frame, handLandmarks)
                gesture = self.recognize_gesture(handLandmarks)
                currentHandLandmarks = handLandmarks
                cv2.putText(frame, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.processGame(gesture, currentHandLandmarks, marker_corners_map, frame.shape)

        calib_status = "OK" if self.homography_matrix is not None else "SEARCHING CORNERS"
        cv2.putText(
            frame,
            f"Game State: {self.gameState} | Arena: {calib_status}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("camera", frame)
        cv2.waitKey(1)

    def drawLandmarks(self, frame, handLandmarks):
        h, w, _ = frame.shape
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in handLandmarks]
        for p1, p2 in connections:
            cv2.line(frame, coords[p1], coords[p2], (0, 255, 0), 2)
        for x, y in coords:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def recognize_gesture(self, handLandmarks):
        fingerTips = [8, 12, 16, 20]
        fingerMcps = [5, 9, 13, 17]
        wrist = handLandmarks[0]

        extendedFingers = 0
        for tipIdx, mcpIdx in zip(fingerTips, fingerMcps):
            tip = handLandmarks[tipIdx]
            mcp = handLandmarks[mcpIdx]

            distTip = np.hypot(tip.x - wrist.x, tip.y - wrist.y)
            distMcp = np.hypot(mcp.x - wrist.x, mcp.y - wrist.y)

            if distTip > distMcp:
                extendedFingers += 1

        if extendedFingers >= 4:
            return "PALM"
        elif extendedFingers <= 1:
            return "FIST"

        return "UNKNOWN"

    def isHandNearCup(self, handLandmarks, marker_corners_map, frameShape, distanceThreshold=170.0):
        if not handLandmarks or self.cup_held:
            return False

        h, w, _ = frameShape
        wrist = handLandmarks[0]
        wristPx = np.array([wrist.x * w, wrist.y * h], dtype=np.float32)

        if self.cupID in marker_corners_map:
            pts = marker_corners_map[self.cupID][0]
            targetPx = np.mean(pts, axis=0)
            dist = np.hypot(wristPx[0] - targetPx[0], wristPx[1] - targetPx[1])
            return dist <= distanceThreshold

        if self.lastCupPixelPos is not None:
            dist = np.hypot(wristPx[0] - self.lastCupPixelPos[0], wristPx[1] - self.lastCupPixelPos[1])
            return dist <= distanceThreshold

        return False

    def processGame(self, gesture, handLandmarks=None, marker_corners_map=None, frameShape=(480, 640, 3)):
        arenaReady = self.homography_matrix is not None
        cupDetected = self.cupID in marker_corners_map and not self.cup_held
        handDetected = handLandmarks is not None and len(handLandmarks) > 0
        currentTime = time.time()

        if self.gameState == "Human":
            if not self.cooldownActive and handDetected and gesture in ("PALM", "FIST"):
                near_cup = self.isHandNearCup(handLandmarks, marker_corners_map, frameShape)

                if cupDetected:
                    if gesture == "PALM" and near_cup and arenaReady:
                        self.get_logger().info("VALID MOVE: PALM on cup detected.")
                        self.cooldownActive = True
                        self.cooldownStartTime = currentTime
                        self.tempo = random.uniform(1.0, 3.0)
                        self._publish_tempo()
                        self.get_logger().info(f"Cooldown active ({self.tempo:.2f}s). Awaiting hand removal...")

                else:
                    if gesture == "FIST":
                        self.get_logger().info("VALID MOVE: FIST on empty space detected.")
                        self.cooldownActive = True
                        self.cooldownStartTime = currentTime
                        self.tempo = random.uniform(1.0, 3.0)
                        self._publish_tempo()
                        self.get_logger().info(f"Cooldown active ({self.tempo:.2f}s). Awaiting hand removal...")

            elif self.cooldownActive:
                elapsed = currentTime - self.cooldownStartTime
                if not handDetected and elapsed >= self.tempo:
                    self.get_logger().info("Hand removed and tempo elapsed. Transitioning to Robot turn.")
                    self.cooldownActive = False
                    self.gameState = "Robot"

        elif self.gameState == "Robot":
            if self.cup_held:
                self.get_logger().info('Robot holding cup. Publishing "place" action command.')
                self.action_pub.publish(String(data="place"))
                self.arm_executing = True
                self.cooldownStartTime = currentTime
                self.gameState = "RobotExecuting"

            elif arenaReady and cupDetected and self.isCupStable():
                x, y = self.cupHistory[-1]
                point_msg = Point(x=float(x), y=float(y), z=0.02)
                self.get_logger().info(f"Publishing homography target -> x: {x:.3f}m, y: {y:.3f}m")
                self.robot_pub.publish(point_msg)
                self.arm_executing = True
                self.cooldownStartTime = currentTime
                self.gameState = "RobotExecuting"

            elif arenaReady and len(self.cupHistory) > 0:
                avgCoords = np.mean(self.cupHistory, axis=0)
                point_msg = Point(x=float(avgCoords[0]), y=float(avgCoords[1]), z=0.02)
                self.get_logger().info(f"Publishing averaged homography target -> x: {point_msg.x:.3f}m, y: {point_msg.y:.3f}m")
                self.robot_pub.publish(point_msg)
                self.arm_executing = True
                self.cooldownStartTime = currentTime
                self.gameState = "RobotExecuting"

            elif not arenaReady:
                self.get_logger().warn("Arena markers (IDs 2, 3, 4, 5) not all detected. Awaiting homography update.")
            else:
                self.get_logger().warn("Robot turn active, but no valid cup position available.")

        elif self.gameState == "RobotExecuting":
            if currentTime - self.cooldownStartTime > 20.0:
                self.get_logger().warn("Safety timeout reached in RobotExecuting. Forcing state back to Human.")
                self.arm_executing = False
                self.cooldownActive = False
                self.gameState = "Human"

    def isCupStable(self):
        """Check if cup coordinates have stabilized within cupThreshold over historyLength frames."""
        if len(self.cupHistory) < self.historyLength:
            return False
        coord = np.array(self.cupHistory)
        std_dev = np.std(coord, axis=0)
        return float(std_dev[0]) < self.cupThreshold and float(std_dev[1]) < self.cupThreshold


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()