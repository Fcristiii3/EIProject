import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import time
import random
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()

        self.robot_pub = self.create_publisher(Point, '/comput_game/camera_coordinates', 10)

        self.tempo = random.uniform(1.0, 3.0)
        self.cooldownActive = False
        self.cooldownStartTime = 0.0

        try:
            calibData = np.load("camera_calibration.npz")
        except FileNotFoundError:
            self.get_logger().error("Camera calibration file not found ('camera_calibration.npz')")
            return

        self.camera_matrix = calibData['camera_matrix']
        self.dist_coeff = calibData['dist_coeff']
        self.robotBaseID = 0
        self.cupID = 1

        # ArUco marker configuration
        self.MARKER_SIZE = 0.06
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        baseOptions = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=baseOptions,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cupHistory = []
        self.historyLength = 5
        self.cupThreshold = 0.015 
        self.lastCupPixelPos = None

        self.gameState = "Human"

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow("camera")
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        for _ in range(2):
            self.cap.grab()

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("No frames captured from camera.")
            return

        corners, ids, _ = self.detector.detectMarkers(frame)
        poses = {}
        marker_corners_map = {}

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for idx, marker_id in enumerate(ids.flatten()):
                marker_id_int = int(marker_id)
                obj_pts = np.array([
                    [-self.MARKER_SIZE/2,  self.MARKER_SIZE/2, 0],
                    [ self.MARKER_SIZE/2,  self.MARKER_SIZE/2, 0],
                    [ self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0],
                    [-self.MARKER_SIZE/2, -self.MARKER_SIZE/2, 0]
                ], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, corners[idx], self.camera_matrix, self.dist_coeff
                )
                if success:
                    poses[marker_id_int] = tvec.flatten()
                    marker_corners_map[marker_id_int] = corners[idx]

        # Continuously record cup relative pose and last known pixel coordinates
        if self.robotBaseID in poses and self.cupID in poses:
            base = poses[self.robotBaseID]
            cup = poses[self.cupID]
            rel_x = float(cup[0] - base[0])
            rel_y = float(cup[1] - base[1])

            self.cupHistory.append((rel_x, rel_y))
            if len(self.cupHistory) > self.historyLength:
                self.cupHistory.pop(0)

            pts = marker_corners_map[self.cupID][0]
            self.lastCupPixelPos = np.mean(pts, axis=0)

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

        self.processGame(poses, gesture, currentHandLandmarks, marker_corners_map, frame.shape)

        cv2.putText(frame, f"Game State: {self.gameState}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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
            (5, 9), (9, 13), (13, 17)
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
        if not handLandmarks:
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

    def processGame(self, poses, gesture, handLandmarks=None, marker_corners_map=None, frameShape=(480, 640, 3)):
        baseDetected = self.robotBaseID in poses
        cupDetected = self.cupID in poses
        handDetected = handLandmarks is not None and len(handLandmarks) > 0
        currentTime = time.time()

        if self.gameState == "Human":
            if not self.cooldownActive:
                if gesture == "PALM" and baseDetected:
                    if self.isHandNearCup(handLandmarks, marker_corners_map, frameShape):
                        self.get_logger().info("PALM gesture near cup detected.")
                        self.cooldownActive = True
                        self.cooldownStartTime = currentTime
                        self.tempo = random.uniform(1.0, 3.0)
                        self.get_logger().info(f"Cooldown active ({self.tempo:.2f}s). Awaiting hand removal...")

                elif gesture == "FIST" and baseDetected:
                    if self.isHandNearCup(handLandmarks, marker_corners_map, frameShape):
                        self.get_logger().info("FIST gesture near cup detected.")
                        self.cooldownActive = True
                        self.cooldownStartTime = currentTime
                        self.tempo = random.uniform(1.0, 3.0)
                        self.get_logger().info(f"Cooldown active ({self.tempo:.2f}s). Awaiting hand removal...")

            else:
                elapsed = currentTime - self.cooldownStartTime
                if not handDetected and elapsed >= self.tempo:
                    self.get_logger().info("Hand removed and tempo elapsed. Transitioning to Robot turn.")
                    self.cooldownActive = False
                    self.gameState = "Robot"

        elif self.gameState == "Robot":
            if baseDetected and cupDetected and self.isCupStable():
                x, y = self.cupHistory[-1]

                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                point_msg.z = 0.02

                self.get_logger().info(f"Publishing {x} and {y}")
                self.robot_pub.publish(point_msg)

                self.cooldownActive = True
                self.cooldownStartTime = currentTime
                self.gameState = "RobotExecuting"

            elif baseDetected and len(self.cupHistory) > 0:
                avgCoords = np.mean(self.cupHistory, axis=0)
                x = float(avgCoords[0])
                y = float(avgCoords[1])

                point_msg = Point()
                point_msg.x = float(x)
                point_msg.y = float(y)
                point_msg.z = 0.1

                self.get_logger().info(f"Publishing {x} and {y}")
                self.robot_pub.publish(point_msg)

                self.cooldownActive = True
                self.cooldownStartTime = currentTime
                self.gameState = "RobotExecuting"

            elif not baseDetected:
                self.get_logger().warn("Base marker (ID 0) missing. Cannot evaluate robot-relative pose.")
            else:
                self.get_logger().warn("Robot turn active, but no valid cup position or history available.")

        elif self.gameState == "RobotExecuting":
            elapsed = currentTime - self.cooldownStartTime
            if elapsed >= self.tempo:
                self.get_logger().info("Robot move duration complete. Human turn starts!")
                self.cooldownActive = False
                self.gameState = "Human"

    def isCupStable(self):
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

if __name__ == '__main__':
    main()