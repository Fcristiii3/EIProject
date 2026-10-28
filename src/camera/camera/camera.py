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

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.robot_pub = self.create_publisher(Point, '/robot/pose', 10)
        

        self.paperHeight = 297
        self.paperWidth = 210
        self.pointsClicked=[]
        self.calibrated=False
        self.cupColorLow = None
        self.cupColorHigh = None
        self.H = None

        self.get_logger().info("click 4 points for paper corners, 1 for arm base, 1 for cup color")

        cv2.namedWindow("initialization")
        cv2.setMouseCallback("initialization", self.clickCorners)

    def clickCorners(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibrated:
            self.pointsClicked.append((x, y))
            pointsLength = len(self.pointsClicked)
            self.get_logger().info(f"Point clicked: ({x}, {y}) for the {pointsLength} point")
            if len(self.pointsClicked) == 6:
                pass

    def calibrateCoords(self, frame):
        paper_points = np.array(self.pointsClicked[:4], dtype=np.float32)
        robot_points = np.array(self.pointsClicked[4], dtype=np.float32)
        cup_points = np.array(self.pointsClicked[5], dtype=np.float32)

        paper_size = np.array([[0, 0], [self.paperWidth, 0], [self.paperWidth, self.paperHeight], [0, self.paperHeight]], dtype=np.float32)
        H_paper = cv2.getPerspectiveTransform(paper_points, paper_size)
        robot_local = np.dot(H_paper, np.array([[x, y, 1] for x, y in robot_points]).T)
        robotX=float(robot_local[0])
        robotY=float(robot_local[1])    
        
        mappedRobotPoints = paper_size - np.array([[robotX, robotY]], dtype=np.float32)
        self.H, _ = cv2.findHomography(paper_points, mappedRobotPoints)

        cx, cy = cup_points
        poi = frame[cy-2:cy+3, cx-2:cx+3]
        avg_color = np.mean(poi, axis=(0, 1))
        self.cupColorLow = np.clip(avg_color - 40, 0, 255).astype(np.uint8)
        self.cupColorHigh = np.clip(avg_color + 40, 0, 255).astype(np.uint8)
        self.calibrated = True

    def processFrame(self, frame):
        #threshold
        mask = cv2.inRange(frame, self.cupColorLow, self.cupColorHigh)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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

                    robotX = world_coords[0] / world_coords[2]
                    robotY = world_coords[1] / world_coords[2]
                    self.get_logger().info(f"Robot Coordinates: ({robotX[0]}, {robotY[0]})")

                    msg = Point()
                    msg.position.x = robotX[0]
                    msg.position.y = robotY[0]
                    msg.position.z = 0.0
                    self.robot_pub.publish(msg)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if not self.calibrated:
            cv2.imshow("initialization", frame)
            cv2.waitKey(1)
            if len(self.pointsClicked) == 6:
                self.calibrateCoords(frame)
                cv2.destroyWindow("initialization")

            for i, point in enumerate(self.pointsClicked):
                if i < 4:
                    color=(0, 255, 0)  
                elif i == 4:
                    color=(255, 0, 0)
                else:
                    color=(0, 0, 255)
                cv2.circle(frame, point, 5, color, -1)
        
        else:
            self.processFrame(frame)
            cv2.imshow("Processed Frame", frame)
            cv2.waitKey(1)
def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    rclpy.spin(camera_node)
    camera_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()