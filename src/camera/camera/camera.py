import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()

        self.robot_pub = self.create_publisher(Point, '/robot/pose', 10)
        
        self.paperHeight = 297
        self.paperWidth = 210
        self.pointsClicked=[]
        self.calibrated=False
        self.cupColorLow = None
        self.cupColorHigh = None
        self.H = None

        self.get_logger().info("click 4 points for paper corners, 1 for arm base, 1 for cup color")

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow("camera")
        cv2.setMouseCallback("camera", self.clickCorners)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("no frames captured")
            return
        frameNoPoint = frame.copy()
        if not self.calibrated:
            for i, point in enumerate(self.pointsClicked):
                if i < 4:
                    color=(0, 255, 0)  
                elif i == 4:
                    color=(255, 0, 0)
                else:
                    color=(0, 0, 255)
                cv2.circle(frame, point, 5, color, -1)
            if len(self.pointsClicked) == 6:
                self.calibrateCoords(frameNoPoint)
        else:
            self.processFrame(frame)
        cv2.imshow("camera", frame)
        cv2.waitKey(1)

    def clickCorners(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibrated:
            if len(self.pointsClicked) < 6:
                self.pointsClicked.append((x, y))
                pointsLength = len(self.pointsClicked)
                self.get_logger().info(f"Point clicked: ({x}, {y}) for the {pointsLength} point")


    def calibrateCoords(self, frame):
        paper_points = np.array(self.pointsClicked[:4], dtype=np.float32)
        robot_points = np.array(self.pointsClicked[4], dtype=np.float32)
        cup_points = np.array(self.pointsClicked[5], dtype=np.float32)

        paper_size = np.array([[0, 0], [self.paperWidth, 0], [self.paperWidth, self.paperHeight], [0, self.paperHeight]], dtype=np.float32)
        H_paper = cv2.getPerspectiveTransform(paper_points, paper_size)

        rx, ry = robot_points
        robot_local = np.dot(H_paper, np.array([rx, ry, 1.0], dtype=np.float32).T)
        robotX=float(robot_local[0])
        robotY=float(robot_local[1])    
        
        mappedRobotPoints = paper_size - np.array([[robotX, robotY]], dtype=np.float32)
        self.H, _ = cv2.findHomography(paper_points, mappedRobotPoints)

        cx, cy = int(cup_points[0]), int(cup_points[1])
        poi = frame[max(0, cy-2):cy+3, max(0, cx-2):cx+3]
        avg = np.mean(poi, axis=(0, 1))

        self.get_logger().info(f"h:{avg[0]}, s:{avg[1]}, v:{avg[2]}")

        self.cupColorLow = np.array([max(0, avg[0]-10), max(0, avg[1]-50), max(0, avg[2]-50)], dtype=np.uint8)
        self.cupColorHigh = np.array([min(255, avg[0]+10), min(255, avg[1]+50), min(255, avg[2]+50)], dtype=np.uint8)

        self.get_logger().info(f"cup color low: {self.cupColorLow}, cup color high: {self.cupColorHigh}")

        self.calibrated = True
        self.get_logger().info("calibration complete")

    def processFrame(self, frame):
        
        mask = cv2.inRange(frame, self.cupColorLow, self.cupColorHigh)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        #self.get_logger().info("entered here")
        #mask = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        cv2.imshow("mask", mask)
        cv2.waitKey(1)


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 200:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"({cx}, {cy})", (int(cx) + 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    pixels=np.array([cx, cy, 1.0], dtype=np.float32).reshape(3, 1)
                    world_coords = np.dot(self.H, pixels)
                    self.get_logger().info(f"World Coordinates: ({world_coords[0][0]}, {world_coords[1][0]})")

                    robotX = world_coords[0] / world_coords[2]
                    robotY = world_coords[1] / world_coords[2]
                    self.get_logger().info(f"Robot Coordinates: ({robotX[0]}, {robotY[0]})")

                    msg = Point()
                    msg.x = robotX[0]/1000.0
                    msg.y = robotY[0]/1000.0
                    msg.z = 0.0
                    self.robot_pub.publish(msg)
                    self.get_logger().info(f"cup position relative to robot:({robotX[0]/1000.0}, {robotY[0]/1000.0}, {msg.z})")


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()