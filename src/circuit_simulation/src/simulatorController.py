import numpy as np
import random
import rospy
import time
import cv2
import sys


from sensor_msgs.msg import Image as ImageCamera
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class ObjectDetector:
    def __init__(self):
        self.darknetSub = rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknetCallback)
        self.boundingBoxes = []

    def getObjects(self):
        objectsDetected = []
        for boundingBox in self.boundingBoxes:
            if boundingBox.Class == "stop sign":
                objectsDetected.append(boundingBox.Class)

        return objectsDetected

    def getBoundingBoxes(self):
        boundingBoxes = self.boundingBoxes
        if random.randrange(0, 9) == 1:
            self.boundingBoxes = []  # ??

        return boundingBoxes

    def darknetCallback(self, msg):
        self.boundingBoxes = msg.bounding_boxes


class Camera:
    def __init__(self, topic, layout, layoutCamera, poseWidget):
        self.topic = topic
        self.image = np.zeros((3, 3, 3),
                              np.uint8)
        self.imageWidth = 320
        self.imageHeight = 240
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(
            self.topic, ImageCamera, self.cameraCallback)
        self.layoutCamera = layoutCamera
        self.layout = layout
        self.poseWidget = poseWidget

    def update(self, boundingBoxes):  # Update camera
        for boundingBox in boundingBoxes:
            if boundingBox.Class == "stop sign" or boundingBox.Class == "traffic light":
                # print("Stop detected")
                cv2.rectangle(self.image, (boundingBox.xmin, boundingBox.ymin), (
                    boundingBox.xmax, boundingBox.ymax), (0, 255, 0), 2)
                cv2.putText(self.image, boundingBox.Class, (
                    boundingBox.xmin, boundingBox.ymin - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        imageResize = cv2.resize(
            self.image, (self.imageWidth, self.imageHeight))
        image = QImage(
            imageResize.data, imageResize.shape[1], imageResize.shape[0], imageResize.shape[1] * imageResize.shape[2], QImage.Format_RGB888)
        self.layoutCamera.setPixmap(QPixmap.fromImage(image))
        self.layout.addWidget(self.layoutCamera, 9, self.poseWidget)

    def cameraCallback(self, img):  # Camera Callback
        self.image = self.bridge.imgmsg_to_cv2(img, "rgb8")


class AppUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.window = QWidget()
        self.layout = QGridLayout()
        self.simulatorControllerLabel = QLabel('Simulator Controller')
        self.simulatorControllerLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.simulatorControllerLabel, 1, 0, 1, 3)
        self.forwardButton = QToolButton()
        self.forwardButton.setArrowType(Qt.UpArrow)
        self.forwardButton.setFixedWidth(75)
        self.forwardButton.clicked.connect(self.goForward)
        self.layout.addWidget(self.forwardButton, 10, 1)
        self.backwardButton = QToolButton()
        self.backwardButton.setFixedWidth(75)
        self.backwardButton.setArrowType(Qt.DownArrow)
        self.backwardButton.clicked.connect(self.goBackward)
        self.layout.addWidget(self.backwardButton, 12, 1)

        self.rightButton = QToolButton()
        self.rightButton.setFixedWidth(75)
        self.rightButton.setArrowType(Qt.RightArrow)
        self.rightButton.clicked.connect(self.goRight)
        self.layout.addWidget(self.rightButton, 11, 2)

        self.leftButton = QToolButton()
        self.leftButton.setFixedWidth(75)
        self.leftButton.setArrowType(Qt.LeftArrow)
        self.leftButton.clicked.connect(self.goLeft)
        self.layout.addWidget(self.leftButton, 11, 0)

        self.stopButton = QToolButton()
        self.stopButton.setFixedWidth(75)
        self.stopButton.setArrowType(Qt.NoArrow)
        self.stopButton.clicked.connect(self.stop)
        self.layout.addWidget(self.stopButton, 11, 1)

        self.startSimulationButton = QPushButton('Start Simulation')
        self.layout.addWidget(self.startSimulationButton, 1, 3)
        self.startSimulationButton.clicked.connect(self.startSimulation)

        self.restartSimulationButton = QPushButton('Restart Simulation')
        self.layout.addWidget(self.restartSimulationButton, 12, 3)
        self.restartSimulationButton.clicked.connect(self.restartSimulation)

        self.centralWidget = QWidget()
        self.centralWidget.setObjectName("centralwidget")
        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)
        self.layoutWidget = QWidget(self.centralWidget)

        self.objectDetector = ObjectDetector()

        self.layoutCamera = QLabel(self.layoutWidget)
        self.camera = Camera("/camera/image_raw",
                             self.layout, self.layoutCamera, 3)
        self.camera.update(self.objectDetector.getBoundingBoxes())

        rospy.init_node("simulator_controller")
        self.cameraButton = QPushButton("Active Camera")
        self.layout.addWidget(self.cameraButton, 10, 3)

        self.cameraButton.setCheckable(True)
        self.cameraButton.clicked.connect(self.onPress)
        self.cameraTimer = QTimer()
        self.cameraTimer.timeout.connect(self.updateCameraTimer)

        self.simulationTimer = QTimer()
        self.simulationTimer.timeout.connect(self.executeSimulation)

        # 10 ??????????
        # Node frequency ??
        self.pubFrontRight = rospy.Publisher(
            '/rover/joint_wheel_back_right_velocity_controller/command', Float64, queue_size=10)
        self.pubFrontLeft = rospy.Publisher(
            '/rover/joint_wheel_back_left_velocity_controller/command', Float64, queue_size=10)
        self.pubBackRight = rospy.Publisher(
            '/rover/joint_wheel_front_right_velocity_controller/command', Float64, queue_size=10)
        self.pubBackLeft = rospy.Publisher(
            '/rover/joint_wheel_front_left_velocity_controller/command', Float64, queue_size=10)
        self.pubRestart = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)

        self.subStateRobot = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.getStateRobot)

        self.time = 0
        self.timeStop = 0
        self.simulationStarted = False
        self.stopSignDetected = False
        self.stopSignTime = False

    def onPress(self):
        self.cameraTimer.start(1)
        self.cameraButton.setCheckable(False)

    # Update camera while camera button is pressed
    def updateCameraTimer(self):
        self.camera.update(self.objectDetector.getBoundingBoxes())

    def getStateRobot(self, msg):
        for i in range(0, len(msg.name)):
            if msg.name[i] == "rover":
                self.robotOrientation = round(msg.pose[i].orientation.w, 2)
                # print(self.robotOrientation)

    def executeSimulation(self):
        objectsDetected = self.objectDetector.getObjects()

        if "stop sign" in objectsDetected and not self.stopSignDetected and not self.stopSignTime:
            self.stopSignDetected = True
            self.stop()
            self.timeStop = time.time()
            print("Stop due to stop sign")
        elif self.stopSignDetected:
            if time.time() - self.timeStop > 5:
                self.stopSignDetected = False
                self.stopSignTime = True
                print("self.time -= 5" + str(self.time) +
                      " " + str(self.time - (time.time() - self.timeStop)))
                self.time = self.time + (time.time() - self.timeStop)
        elif self.stopSignTime and time.time() - self.timeStop > 10:
            self.stopSignTime = False
        elif not self.stopSignDetected:
            if time.time() - self.time > 102:
                self.stop()
                print("Finished")
                sys.exit()
            elif time.time() - self.time > 83 and self.robotOrientation <= -0.7:
                self.goForward()
            elif time.time() - self.time > 81:
                self.goRight()
            elif time.time() - self.time > 78 and self.robotOrientation <= -0.25:
                self.goForward()
            elif time.time() - self.time > 76:
                self.goRight()
            elif time.time() - self.time > 52 and self.robotOrientation <= 0.01:
                self.goForward()
            elif time.time() - self.time > 50:
                self.goRight()
            elif time.time() - self.time > 48 and self.robotOrientation <= 0.4:
                self.goForward()
            elif time.time() - self.time > 46:
                self.goRight()
            elif time.time() - self.time > 28 and self.robotOrientation <= 0.7:
                self.goForward()
            elif time.time() - self.time > 27.5:
                self.goRight()
            elif self.robotOrientation < 0.93:
                self.goForward()
            elif time.time() - self.time > 21:
                self.goRight()
            elif time.time() - self.time > 0:
                self.goForward()

    def startSimulation(self):
        if not self.simulationStarted:
            self.time = time.time()
            self.simulationTimer.start(10)
            self.startSimulationButton.setCheckable(False)
            self.simulationStarted = True

    def restartSimulation(self):
        stateMsg = ModelState()
        stateMsg.model_name = 'rover'
        stateMsg.pose.position.x = -10.0
        stateMsg.pose.position.y = 45.0
        stateMsg.pose.position.z = 4.0
        self.pubRestart.publish(stateMsg)

    def goForward(self):
        msg = Float64()
        msg.data = 5.0
        # print("Forward")
        self.pubFrontRight.publish(msg)
        self.pubFrontLeft.publish(msg)
        self.pubBackRight.publish(msg)
        self.pubBackLeft.publish(msg)

    def goBackward(self):
        msg = Float64()
        msg.data = -3.0
        # print("Backward")
        self.pubFrontRight.publish(msg)
        self.pubFrontLeft.publish(msg)
        self.pubBackRight.publish(msg)
        self.pubBackLeft.publish(msg)

    def goRight(self):
        msg = Float64()
        msg.data = -2
        msg2 = Float64()
        msg2.data = 2
        # print("Right")
        self.pubFrontRight.publish(msg)
        self.pubFrontLeft.publish(msg2)
        self.pubBackRight.publish(msg)
        self.pubBackLeft.publish(msg2)

    def goLeft(self):
        msg = Float64()
        msg.data = -2
        msg2 = Float64()
        msg2.data = 2
        # print("Left")
        self.pubFrontRight.publish(msg2)
        self.pubFrontLeft.publish(msg)
        self.pubBackRight.publish(msg2)
        self.pubBackLeft.publish(msg)

    def stop(self):
        msg = Float64()
        msg.data = 0
        # print("Stop")
        self.pubFrontRight.publish(msg)
        self.pubFrontLeft.publish(msg)
        self.pubBackRight.publish(msg)
        self.pubBackLeft.publish(msg)


def darkMode(app):
    """Apply dark mode to Application"""
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(15, 15, 15))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(0, 87, 184).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)


app = QApplication([])
darkMode(app)
myAPP = AppUI()
myAPP.show()
sys.exit(app.exec_())
