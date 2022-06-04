import numpy as np
import random
import rospy
import time
import cv2
import sys

from sensor_msgs.msg import Image as ImageCamera
from darknet_ros_msgs.msg import BoundingBoxes
from gazebo_msgs.msg import ModelStates, ModelState
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from qt_material import apply_stylesheet


class MapCamera:
    def __init__(self, topic, gridLayout, layoutMapCamera):
        self.imageQT = np.zeros((3, 3, 3),
                                np.uint8)
        self.imageWidth = 320
        self.imageHeight = 240
        self.bridge = CvBridge()
        rospy.Subscriber(
            topic, ImageCamera, self.imageCallback)
        self.layoutMapCamera = layoutMapCamera
        self.gridLayout = gridLayout

    def update(self):  # Update map camera
        imageResized = cv2.resize(
            self.imageQT, (self.imageWidth, self.imageHeight))
        image = QImage(
            imageResized.data, imageResized.shape[1], imageResized.shape[0], imageResized.shape[1] * imageResized.shape[2], QImage.Format_BGR888)
        self.layoutMapCamera.setPixmap(QPixmap.fromImage(image))
        self.gridLayout.addWidget(self.layoutMapCamera, 3, 0)

    def imageCallback(self, img):  # Map camera callback
        self.imageQT = self.bridge.imgmsg_to_cv2(img, "bgr8")


class OnboardCamera:
    def __init__(self, topic, gridLayout, layoutOnboardCamera):
        self.imageQT = np.zeros((3, 3, 3),
                                np.uint8)
        self.imageWidth = 320
        self.imageHeight = 240
        # self.croppedImage = np.zeros(
        #     [self.imageWidth, self.imageHeight, 3], dtype=np.uint8)
        self.bridge = CvBridge()
        rospy.Subscriber(
            topic, ImageCamera, self.imageCallback)
        # self.imageWithObjectsPublisher = rospy.Publisher(
        #     "/image_with_objects", ImageCamera, queue_size=10)
        self.layoutOnboardCamera = layoutOnboardCamera
        self.gridLayout = gridLayout

    def getCroppedImage(self, boundingBoxes):  # Get traffic light cropped image
        for boundingBox in boundingBoxes:
            if boundingBox.Class == "traffic light":
                return self.imageQT[boundingBox.ymin - 10:boundingBox.ymax + 10,
                                    boundingBox.xmin - 7:boundingBox.xmax + 7]

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

        # imageWithObjectsMsg = self.bridge.cv2_to_imgmsg(imageResized, "bgr8")
        # self.imageWithObjectsPublisher.publish(imageWithObjectsMsg)
        image = QImage(
            imageResized.data, imageResized.shape[1], imageResized.shape[0], imageResized.shape[1] * imageResized.shape[2], QImage.Format_BGR888)
        self.layoutOnboardCamera.setPixmap(QPixmap.fromImage(image))
        self.gridLayout.addWidget(self.layoutOnboardCamera, 3, 4)

    def imageCallback(self, img):  # Camera onboard callback
        self.imageQT = self.bridge.imgmsg_to_cv2(img, "bgr8")


class ObjectDetector:
    def __init__(self, gridLayout, layoutOnboardCamera):
        rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknetCallback)
        self.onboardCamera = OnboardCamera("/onboard_camera/image_raw",
                                           gridLayout, layoutOnboardCamera)
        self.boundingBoxes = []
        self.objectsDetected = []

    def searchRed(self, hsvImage):
        firstMask = cv2.inRange(hsvImage, (0, 170, 170), (10, 255, 255))
        secondMask = cv2.inRange(hsvImage, (170, 170, 170), (180, 255, 255))
        maskRed = firstMask | secondMask
        # cv2.imshow("Red", maskRed)

        if cv2.countNonZero(maskRed) > 0:
            return True

        return False

    def searchGreen(self, hsvImage):
        maskGreen = cv2.inRange(hsvImage, (55, 199, 209), (133, 255, 255))
        # cv2.imshow("Green", maskGreen)

        if cv2.countNonZero(maskGreen) > 0:
            return True

        return False

    def getObjects(self):
        return self.objectsDetected

    def update(self):
        self.onboardCamera.update(self.getBoundingBoxes())

    def getBoundingBoxes(self):
        boundingBoxes = self.boundingBoxes
        if random.randrange(0, 50) == 1:
            self.boundingBoxes = []  # ??

        return boundingBoxes

    def darknetCallback(self, msg):
        self.objectsDetected = []
        self.boundingBoxes = msg.bounding_boxes
        for boundingBox in self.boundingBoxes:
            if boundingBox.Class == "traffic light" and boundingBox.probability >= 0.9:
                try:
                    croppedImage = self.onboardCamera.getCroppedImage(
                        self.getBoundingBoxes())
                    # cv2.imshow("Original", croppedImage)
                    hsvImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)

                    if self.searchRed(hsvImage):
                        boundingBox.Class = "traffic light red"
                    elif self.searchGreen(hsvImage):
                        boundingBox.Class = "traffic light green"
                except:
                    print("An exception occurred")
            elif boundingBox.Class == "stop sign" and boundingBox.probability < 0.97:
                pass
            elif boundingBox.Class == "person" and boundingBox.probability < 0.9:
                pass

            self.objectsDetected.append(boundingBox.Class)


class AppUI(QMainWindow):
    def __init__(self):
        super().__init__()
        gridLayout = QGridLayout()
        centralWidget = QWidget()
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)
        layoutWidget = QWidget(centralWidget)

        layoutOnboardCamera = QLabel(layoutWidget)
        self.objectDetector = ObjectDetector(
            gridLayout, layoutOnboardCamera)

        layoutMapCamera = QLabel(layoutWidget)
        self.mapCamera = MapCamera("/map_camera/image_raw",
                                   gridLayout, layoutMapCamera)

        self.objectDetector.update()
        self.mapCamera.update()

        simulatorControllerButton = QPushButton('Simulator Controller')
        simulatorControllerButton.setProperty('class', 'warning')
        simulatorControllerButton.setCheckable(True)
        simulatorControllerButton.setChecked(True)
        simulatorControllerButton.setFixedWidth(200)
        gridLayout.addWidget(simulatorControllerButton, 0, 0, 3, 0,
                             alignment=Qt.AlignHCenter)

        forwardButton = QToolButton()
        forwardButton.setArrowType(Qt.UpArrow)
        forwardButton.setFixedWidth(75)
        forwardButton.clicked.connect(self.goForward)
        gridLayout.addWidget(forwardButton, 10, 2)
        backwardButton = QToolButton()
        backwardButton.setFixedWidth(75)
        backwardButton.setArrowType(Qt.DownArrow)
        backwardButton.clicked.connect(self.goBackward)
        gridLayout.addWidget(backwardButton, 12, 2)

        rightButton = QToolButton()
        rightButton.setFixedWidth(75)
        rightButton.setArrowType(Qt.RightArrow)
        rightButton.clicked.connect(self.goRight)
        gridLayout.addWidget(rightButton, 11, 3)

        leftButton = QToolButton()
        leftButton.setFixedWidth(75)
        leftButton.setArrowType(Qt.LeftArrow)
        leftButton.clicked.connect(self.goLeft)
        gridLayout.addWidget(leftButton, 11, 1)

        stopButton = QPushButton()
        stopButton.setProperty('class', 'danger')
        stopButton.setFixedWidth(15)
        stopButton.setFixedHeight(15)
        stopButton.clicked.connect(self.stop)
        gridLayout.addWidget(stopButton, 11, 2,
                             alignment=Qt.AlignHCenter)

        startSimulationButton = QPushButton('Start Simulation')
        gridLayout.addWidget(startSimulationButton,
                             2, 4, alignment=Qt.AlignVCenter)

        startSimulationButton.clicked.connect(self.startSimulation)

        restartSimulationButton = QPushButton('Restart Simulation')
        restartSimulationButton.setProperty('class', 'danger')
        gridLayout.addWidget(restartSimulationButton, 10, 4)
        restartSimulationButton.clicked.connect(self.restartSimulation)

        self.cameraButton = QPushButton("Activate Camera")
        self.cameraButton.setProperty('class', 'big_button')
        gridLayout.addWidget(self.cameraButton, 10, 0)

        self.cameraButton.setCheckable(True)
        self.cameraButton.clicked.connect(self.onPress)
        self.cameraTimer = QTimer()
        self.cameraTimer.timeout.connect(self.updateCameraTimer)
        self.simulationTimer = QTimer()

        # Node frequency ?? # 10 ?????????? # One publisher instead of four??
        self.frontRightVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/back_right_wheel_velocity_controller/command', Float64, queue_size=10)
        self.frontLeftVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/back_left_wheel_velocity_controller/command', Float64, queue_size=10)
        self.backRightVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/front_right_wheel_velocity_controller/command', Float64, queue_size=10)
        self.backLeftVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/front_left_wheel_velocity_controller/command', Float64, queue_size=10)
        self.actorPositionPublisher = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)
        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.getStateRobot)

        self.timeStart = 0
        self.timeStop = 0
        self.timeRedStop = 0
        self.simulationStarted = False
        self.stopSignDetected = False
        self.stopSignTime = False
        self.redLightDetected = False

    def onPress(self):
        self.cameraTimer.start(1)
        self.cameraButton.setCheckable(False)

    # Update camera while camera button is pressed
    def updateCameraTimer(self):
        self.objectDetector.update()
        self.mapCamera.update()

    def getStateRobot(self, msg):
        for i in range(0, len(msg.name)):
            if msg.name[i] == "autonomous_vehicle":
                self.robotOrientation = round(msg.pose[i].orientation.z, 2)
                self.robotOrientationW = round(msg.pose[i].orientation.w, 2)

    def executeSimulation(self):
        objectsDetected = self.objectDetector.getObjects()

        if "stop sign" in objectsDetected and not self.stopSignDetected and not self.stopSignTime:
            self.stopSignDetected = True
            self.stop()
            self.timeStop = time.time()
            print("Stop due to stop sign")
        elif "traffic light red" in objectsDetected and not self.redLightDetected:
            self.redLightDetected = True
            self.stop()
            self.timeRedStop = time.time()
            print("Stop due to traffic light red")
        elif "traffic light green" in objectsDetected and self.redLightDetected:
            self.redLightDetected = False
            deleteActorMsg = ModelState()
            deleteActorMsg.model_name = 'actor'
            deleteActorMsg.pose.position.x = 0.0
            deleteActorMsg.pose.position.y = 0.0
            deleteActorMsg.pose.position.z = -100.0
            self.actorPositionPublisher.publish(deleteActorMsg)
            print("Actor deleted")
            print("Green light detected, continue...")
            print("self.timeStart -= 5" + str(self.timeStart) +
                  " " + str(self.timeStart - (time.time() - self.timeRedStop)))
            self.timeStart = self.timeStart + (time.time() - self.timeRedStop)
        elif self.stopSignDetected:
            if time.time() - self.timeStop > 5:
                self.stopSignDetected = False
                self.stopSignTime = True
                print("self.timeStart -= 5" + str(self.timeStart) +
                      " " + str(self.timeStart - (time.time() - self.timeStop)))
                self.timeStart = self.timeStart + (time.time() - self.timeStop)
        elif self.stopSignTime and time.time() - self.timeStop > 40:
            self.stopSignTime = False
            print("40 seconds")
        elif not self.stopSignDetected and not self.redLightDetected:
            if rospy.get_time() - self.timeStart > 153.5:
                self.stop()
                print("Finished")
                time.sleep(2)
                sys.exit()
            elif rospy.get_time() - self.timeStart > 123 and self.robotOrientationW <= 0.0:
                self.goForward()
            elif rospy.get_time() - self.timeStart > 121:
                self.goRight()
            elif rospy.get_time() - self.timeStart > 116 and self.robotOrientation <= -0.85:
                self.goForward()
            elif rospy.get_time() - self.timeStart > 115.5:
                self.goRight()
            elif rospy.get_time() - self.timeStart > 77 and self.robotOrientation < -0.7:
                self.goForward()
            elif rospy.get_time() - self.timeStart > 74:
                self.goRight()
            elif rospy.get_time() - self.timeStart > 71 and self.robotOrientation <= 0.4:
                self.goForward()
            elif rospy.get_time() - self.timeStart > 69:
                self.goRight()
            elif rospy.get_time() - self.timeStart > 42 and self.robotOrientation <= 0.02:
                self.goForward()
            elif rospy.get_time() - self.timeStart > 40:
                self.goRight()
            elif self.robotOrientation < 0.4 and rospy.get_time() - self.timeStart > 34:
                self.goForward()
            elif rospy.get_time() - self.timeStart > 33:
                self.goRight()
            elif rospy.get_time() - self.timeStart > 0:
                self.goForward()

    def startSimulation(self):
        if not self.simulationStarted:
            self.timeStart = rospy.get_time()
            self.simulationTimer.timeout.connect(self.executeSimulation)
            self.simulationTimer.start(10)
            self.simulationStarted = True

    def restartSimulation(self):
        stateMsg = ModelState()
        stateMsg.model_name = 'autonomous_vehicle'
        stateMsg.pose.position.x = -10.0
        stateMsg.pose.position.y = 45.0
        stateMsg.pose.position.z = 4.0
        stateMsg.pose.orientation.z = 0.7
        stateMsg.pose.orientation.w = 0.7
        self.actorPositionPublisher.publish(stateMsg)

    def goForward(self):
        rightVelMsg = Float64()
        rightVelMsg.data = -2
        leftVelMsg = Float64()
        leftVelMsg.data = 2
        self.frontRightVelPublisher.publish(rightVelMsg)
        self.frontLeftVelPublisher.publish(leftVelMsg)
        self.backRightVelPublisher.publish(rightVelMsg)
        self.backLeftVelPublisher.publish(leftVelMsg)

    def goBackward(self):
        rightVelMsg = Float64()
        rightVelMsg.data = 2.0
        leftVelMsg = Float64()
        leftVelMsg.data = -2.0
        self.frontRightVelPublisher.publish(rightVelMsg)
        self.frontLeftVelPublisher.publish(leftVelMsg)
        self.backRightVelPublisher.publish(rightVelMsg)
        self.backLeftVelPublisher.publish(leftVelMsg)

    def goRight(self):
        rightVelMsg = Float64()
        rightVelMsg.data = 0.75
        self.frontRightVelPublisher.publish(rightVelMsg)
        self.frontLeftVelPublisher.publish(rightVelMsg)
        self.backRightVelPublisher.publish(rightVelMsg)
        self.backLeftVelPublisher.publish(rightVelMsg)

    def goLeft(self):
        leftVelMsg = Float64()
        leftVelMsg.data = -0.75
        self.frontRightVelPublisher.publish(leftVelMsg)
        self.frontLeftVelPublisher.publish(leftVelMsg)
        self.backRightVelPublisher.publish(leftVelMsg)
        self.backLeftVelPublisher.publish(leftVelMsg)

    def stop(self):
        stopVelMsg = Float64()
        stopVelMsg.data = 0
        self.frontRightVelPublisher.publish(stopVelMsg)
        self.frontLeftVelPublisher.publish(stopVelMsg)
        self.backRightVelPublisher.publish(stopVelMsg)
        self.backLeftVelPublisher.publish(stopVelMsg)


rospy.init_node("simulator_controller")
app = QApplication([])
apply_stylesheet(app, theme='dark_blue.xml')
appUI = AppUI()
appUI.setMinimumSize(910, 512)
appUI.show()
sys.exit(app.exec_())
