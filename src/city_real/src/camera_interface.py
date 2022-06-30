import numpy as np
import rospy
import cv2
import sys

from sensor_msgs.msg import Image as ImageCamera
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qt_material import apply_stylesheet


class Camera:
    def __init__(self, topic, gridLayout, layoutCamera):
        self.imageQT = np.zeros((3, 3, 3),
                                np.uint8)
        self.IMAGE_WIDTH = rospy.get_param("/interfaceParams/IMAGE_WIDTH")
        self.IMAGE_HEIGHT = rospy.get_param("/interfaceParams/IMAGE_HEIGHT")
        self.TEXT_BBOX = rospy.get_param("/interfaceParams/TEXT_BBOX")
        self.LANE_IMAGE_WIDTH = rospy.get_param(
            "/interfaceParams/LANE_IMAGE_WIDTH")
        self.LANE_IMAGE_HEIGHT = rospy.get_param(
            "/interfaceParams/LANE_IMAGE_HEIGHT")
        self.bridge = CvBridge()
        rospy.Subscriber(
            topic, ImageCamera, self.imageCallback)
        self.layoutCamera = layoutCamera
        self.gridLayout = gridLayout

    def update(self, boundingBoxes, centerImage):
        for boundingBox in boundingBoxes:
            cv2.rectangle(self.imageQT, (boundingBox.xmin, boundingBox.ymin), (
                boundingBox.xmax, boundingBox.ymax), (0, 255, 0), 2)
            cv2.putText(self.imageQT, boundingBox.Class, (
                boundingBox.xmin, boundingBox.ymin - self.TEXT_BBOX), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        imageLane = cv2.resize(
            self.imageQT, (self.LANE_IMAGE_WIDTH, self.LANE_IMAGE_HEIGHT))
        prediction = cv2.circle(
            imageLane, (centerImage, centerImage), 8, (255, 0, 0), 3)
        image = cv2.resize(
            prediction, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        image = QImage(
            image.data, image.shape[1], image.shape[0], image.shape[1] * image.shape[2], QImage.Format_BGR888)
        self.layoutCamera.setPixmap(QPixmap.fromImage(image))
        self.gridLayout.addWidget(self.layoutCamera, 3, 0)

    def imageCallback(self, img):
        self.imageQT = self.bridge.imgmsg_to_cv2(img, "bgr8")


class ObjectDetector:
    def __init__(self, gridLayout, layoutCamera):
        rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknetCallback)
        rospy.Subscriber("/center_image", Int32, self.centerImageCallback)
        self.camera = Camera("/usb_cam/image_raw",
                             gridLayout, layoutCamera)
        self.boundingBoxes = []
        self.centerImage = 0

    def update(self):
        self.camera.update(self.boundingBoxes, self.centerImage)

    def darknetCallback(self, msg):
        self.boundingBoxes = msg.bounding_boxes

    def centerImageCallback(self, msg):
        self.centerImage = msg.data
        rospy.loginfo("Center Image: " + str(msg.data))


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

        self.cameraButton = QPushButton("Activate Camera")
        self.cameraButton.setProperty('class', 'big_button')
        gridLayout.addWidget(self.cameraButton, 8, 0)

        self.cameraButton.setCheckable(True)
        self.cameraButton.clicked.connect(self.onPress)
        self.cameraTimer = QTimer()
        self.cameraTimer.timeout.connect(self.updateCameraTimer)

    def onPress(self):
        rospy.loginfo("Camera activated")
        self.cameraTimer.start(1)
        self.cameraButton.setCheckable(False)

    def updateCameraTimer(self):
        self.objectDetector.update()


rospy.init_node("camera_interface")
app = QApplication([])
apply_stylesheet(app, theme='dark_blue.xml')
appUI = AppUI()
WINDOW_WIDTH = rospy.get_param("/interfaceParams/WINDOW_WIDTH")
WINDOW_HEIGHT = rospy.get_param("/interfaceParams/WINDOW_HEIGHT")
appUI.setMinimumSize(WINDOW_WIDTH, WINDOW_HEIGHT)
appUI.show()
sys.exit(app.exec_())
