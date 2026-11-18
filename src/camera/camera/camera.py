import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from ultralytics import YOLOWorld

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()

        self.robot_pub = self.create_publisher(Point, '/comput_game/camera_coordinates', 10)
        
        self.paperHeight = 297
        self.paperWidth = 210
        self.pointsClicked=[]
        self.calibrated=False
        self.H = None

        self.get_logger().info("yolo")
        self.model = YOLOWorld("yolov8s-worldv2.pt")
        self.model.set_classes(["cup","mug","plastic cup","festival plastic cup","hand","hand palm", "hand back", "person hand", "human hand","water bottle","bottle","plastic bottle"])

        self.frame_count = 0
        self.process_every_n_frames = 2
        self.latest_boxes = []
        

        self.cupHistory =[]
        self.historyLength = 5
        self.cupThreshold = 3.0
        self.gameState = "Human"

        self.get_logger().info("click 4 points for paper corners, 1 for arm base")

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow("camera")
        cv2.setMouseCallback("camera", self.clickCorners)

        self.timer = self.create_timer(0.05, self.timer_callback)    

    def timer_callback(self):

        for _ in range(2):
            self.cap.grab()

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
            if len(self.pointsClicked) == 5:
                self.calibrateCoords(frameNoPoint)
        else:
            self.processFrame(frame)
        cv2.imshow("camera", frame)
        cv2.waitKey(1)

    def clickCorners(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibrated:
            if len(self.pointsClicked) < 5:
                self.pointsClicked.append((x, y))
                pointsLength = len(self.pointsClicked)
                self.get_logger().info(f"clicked {x} {y} for the {pointsLength} point")


    def calibrateCoords(self, frame):
        paper_points = np.array(self.pointsClicked[:4], dtype=np.float32)
        robot_points = np.array(self.pointsClicked[4], dtype=np.float32)

        paper_size = np.array([[0, 0], 
        [self.paperWidth, 0], 
        [self.paperWidth, self.paperHeight], 
        [0, self.paperHeight]], 
        dtype=np.float32)
        
        H_paper = cv2.getPerspectiveTransform(paper_points, paper_size)

        rx, ry = robot_points
        robot_local = np.dot(H_paper, np.array([rx, ry, 1.0], dtype=np.float32).T)
        robotX=float(robot_local[0])
        robotY=float(robot_local[1])    
        
        mappedRobotPoints = paper_size - np.array([[robotX, robotY]], dtype=np.float32)
        self.H, _ = cv2.findHomography(paper_points, mappedRobotPoints)

        self.calibrated = True
        self.get_logger().info("calibration complete")

    def processFrame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames == 0:
            results = self.model(frame, verbose=False, conf=0.2)[0]
            self.latest_boxes = results.boxes

        for box in self.latest_boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label_name = self.model.names[cls_id]
            self.get_logger().info(f"Detected: {label_name} ({conf:.2f})")
        
        

        cupCenter = None
        handFlag = False
        for box in self.latest_boxes:
            label = self.model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if "hand" in label.lower():
                handFlag = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            elif "cup" in label.lower() or "mug" in label.lower() or "bottle" in label.lower():
                cupCenter = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        self.updateGameState(cupCenter, handFlag)
        cv2.putText(frame, f"Game State: {self.gameState}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def updateGameState(self, cupCenter, handFlag):
        if self.gameState == "Human":
            if cupCenter is not None:
                self.cupHistory.append(cupCenter)
                if len(self.cupHistory) > self.historyLength:
                    self.cupHistory.pop(0)
                
                if not handFlag and self.cupAvailable():
                    self.gameState = "Robot"
                    self.get_logger().info("Switching to Robot state")

                    cx,cy = self.cupHistory[-1]
                    pixels = np.array([[cx, cy, 1.0]], dtype=np.float32).reshape(3,1)
                    worldCoord = np.dot(self.H, pixels)
                    robotX = (worldCoord[0] / worldCoord[2])[0]
                    robotY = (worldCoord[1] / worldCoord[2])[0]

                    msg = Point()
                    msg.x = robotX
                    msg.y = robotY
                    msg.z = 0.0
                    self.robot_pub.publish(msg)

                    self.get_logger().info(f"Published robot coordinates: x={robotX}, y={robotY}")
        elif self.gameState == "Robot":
            if handFlag:
                self.gameState = "Human"
                self.get_logger().info("Switching to Human state")
                self.cupHistory.clear()
                

    def cupAvailable(self):
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