import numpy as np
import random
import rospy
import time
import cv2
import sys

from sensor_msgs.msg import Image as ImageCamera
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qt_material import apply_stylesheet


class Camera:
    def __init__(self, topic, gridLayout, layoutCamera):
        self.imageQT = np.zeros((3, 3, 3),
                                np.uint8)
        self.imageWidth = 320
        self.imageHeight = 240
        self.bridge = CvBridge()
        rospy.Subscriber(
            topic, ImageCamera, self.imageCallback)
        self.layoutCamera = layoutCamera
        self.gridLayout = gridLayout

    def update(self, boundingBoxes):  # Update onboard camera
        imageResized = cv2.resize(
            self.imageQT, (self.imageWidth, self.imageHeight))
        for boundingBox in boundingBoxes:
            if boundingBox.Class == "stop sign" or boundingBox.Class == "person" or boundingBox.Class == "car" or boundingBox.Class == "truck":
                cv2.rectangle(imageResized, (boundingBox.xmin, boundingBox.ymin), (
                    boundingBox.xmax, boundingBox.ymax), (0, 255, 0), 2)
                cv2.putText(imageResized, boundingBox.Class, (
                    boundingBox.xmin, boundingBox.ymin - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if boundingBox.Class == "traffic light" or boundingBox.Class == "traffic light green" and boundingBox.Class == "traffic light red":
                cv2.rectangle(imageResized, (boundingBox.xmin - 7, boundingBox.ymin - 7), (
                    boundingBox.xmax + 7, boundingBox.ymax + 7), (0, 255, 0), 2)
                cv2.putText(imageResized, boundingBox.Class, (
                    boundingBox.xmin - 7, boundingBox.ymin - 5 - 7), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        image = QImage(
            imageResized.data, imageResized.shape[1], imageResized.shape[0], imageResized.shape[1] * imageResized.shape[2], QImage.Format_BGR888)
        self.layoutCamera.setPixmap(QPixmap.fromImage(image))
        self.gridLayout.addWidget(self.layoutCamera, 3, 0)

    def imageCallback(self, img):  # Camera onboard callback
        self.imageQT = self.bridge.imgmsg_to_cv2(img, "bgr8")


class ObjectDetector:
    def __init__(self, gridLayout, layoutCamera):
        rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknetCallback)
        self.camera = Camera("/usb_cam/image_raw",
                             gridLayout, layoutCamera)
        self.boundingBoxes = []
        self.objectsDetected = []

    def getObjects(self):
        return self.objectsDetected

    def update(self):
        self.camera.update(self.getBoundingBoxes())

    def getBoundingBoxes(self):
        boundingBoxes = self.boundingBoxes
        if random.randrange(0, 50) == 1:
            self.boundingBoxes = []  # ??

        return boundingBoxes

    def darknetCallback(self, msg):
        self.objectsDetected = []
        self.boundingBoxes = msg.bounding_boxes
        for boundingBox in self.boundingBoxes:
            self.objectsDetected.append(boundingBox.Class)


class AppUI(QMainWindow):
    def __init__(self):
        super().__init__()
        gridLayout = QGridLayout()
        centralWidget = QWidget()
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)
        layoutWidget = QWidget(centralWidget)

        layoutCamera = QLabel(layoutWidget)
        self.objectDetector = ObjectDetector(
            gridLayout, layoutCamera)

        self.objectDetector.update()

        autonomousVehicleUIButton = QPushButton('Autonomous Vehicle UI')
        autonomousVehicleUIButton.setProperty('class', 'warning')
        autonomousVehicleUIButton.setCheckable(True)
        autonomousVehicleUIButton.setChecked(True)
        gridLayout.addWidget(autonomousVehicleUIButton, 0, 0)

        stopCNNButton = QPushButton('Stop')
        stopCNNButton.setProperty('class', 'danger')
        gridLayout.addWidget(stopCNNButton, 8, 0)
        stopCNNButton.clicked.connect(self.stopCNN)

        self.cameraButton = QPushButton("Activate Camera")
        self.cameraButton.setProperty('class', 'big_button')
        gridLayout.addWidget(self.cameraButton, 2, 0)

        self.cameraButton.setCheckable(True)
        self.cameraButton.clicked.connect(self.onPress)
        self.cameraTimer = QTimer()
        self.cameraTimer.timeout.connect(self.updateCameraTimer)
        self.simulationTimer = QTimer()

        self.CNNStarted = False

    def onPress(self):
        print("Activate Camera")
        self.cameraTimer.start(1)
        self.cameraButton.setCheckable(False)

    # Update camera while camera button is pressed
    def updateCameraTimer(self):
        self.objectDetector.update()

    def startCNN(self):
        if not self.CNNStarted:
            print("Start CNN")
            self.CNNStarted = True

    def stopCNN(self):
        print("Stop CNN")

    def stop(self):
        print("stop")


rospy.init_node("autonomous_vehicle_ui")
app = QApplication([])
apply_stylesheet(app, theme='dark_blue.xml')
appUI = AppUI()
appUI.setMinimumSize(310, 512)
appUI.show()
sys.exit(app.exec_())
