import torchvision.transforms as transforms
import numpy as np
import torchvision
import PIL.Image
import random
import torch
import rospy
import time
import cv2
import sys

from gazebo_msgs.msg import ModelStates, ModelState
from sensor_msgs.msg import Image as ImageCamera
from darknet_ros_msgs.msg import BoundingBoxes
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
        self.IMAGE_WIDTH = rospy.get_param("/simParams/IMAGE_WIDTH")
        self.IMAGE_HEIGHT = rospy.get_param("/simParams/IMAGE_HEIGHT")
        self.bridge = CvBridge()
        rospy.Subscriber(
            topic, ImageCamera, self.imageCallback)
        self.layoutMapCamera = layoutMapCamera
        self.gridLayout = gridLayout

    def update(self):
        imageResized = cv2.resize(
            self.imageQT, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        image = QImage(
            imageResized.data, imageResized.shape[1], imageResized.shape[0], imageResized.shape[1] * imageResized.shape[2], QImage.Format_BGR888)
        self.layoutMapCamera.setPixmap(QPixmap.fromImage(image))
        self.gridLayout.addWidget(self.layoutMapCamera, 3, 0)

    def imageCallback(self, img):
        self.imageQT = self.bridge.imgmsg_to_cv2(img, "bgr8")


class OnboardCamera:
    def __init__(self, topic):
        self.imageQT = np.zeros((3, 3, 3),
                                np.uint8)
        self.IMAGE_WIDTH = rospy.get_param("/simParams/IMAGE_WIDTH")
        self.IMAGE_HEIGHT = rospy.get_param("/simParams/IMAGE_HEIGHT")
        self.bridge = CvBridge()
        rospy.Subscriber(
            topic, ImageCamera, self.imageCallback)
        self.INCREMENT_BBOX_X = rospy.get_param("/simParams/INCREMENT_BBOX_X")
        self.INCREMENT_BBOX_Y = rospy.get_param("/simParams/INCREMENT_BBOX_Y")
        self.TEXT_BBOX = rospy.get_param("/simParams/TEXT_BBOX")

    def getCroppedImage(self, boundingBoxes):
        for boundingBox in boundingBoxes:
            if boundingBox.Class == "traffic light":
                return self.imageQT[boundingBox.ymin - self.INCREMENT_BBOX_Y:boundingBox.ymax + self.INCREMENT_BBOX_Y,
                                    boundingBox.xmin - self.INCREMENT_BBOX_X:boundingBox.xmax + self.INCREMENT_BBOX_X]

    def update(self, boundingBoxes):
        imageResized = cv2.resize(
            self.imageQT, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        for boundingBox in boundingBoxes:
            if boundingBox.Class == "stop sign" or boundingBox.Class == "person" or boundingBox.Class == "car" or boundingBox.Class == "truck":
                cv2.rectangle(imageResized, (boundingBox.xmin, boundingBox.ymin), (
                    boundingBox.xmax, boundingBox.ymax), (0, 255, 0), 2)
                cv2.putText(imageResized, boundingBox.Class, (
                    boundingBox.xmin, boundingBox.ymin - self.TEXT_BBOX), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            if boundingBox.Class == "traffic light" or boundingBox.Class == "traffic light green" and boundingBox.Class == "traffic light red":
                cv2.rectangle(imageResized, (boundingBox.xmin - self.INCREMENT_BBOX_X, boundingBox.ymin - self.INCREMENT_BBOX_Y), (
                    boundingBox.xmax + self.INCREMENT_BBOX_X, boundingBox.ymax + self.INCREMENT_BBOX_Y), (0, 255, 0), 2)
                cv2.putText(imageResized, boundingBox.Class, (
                    boundingBox.xmin - self.INCREMENT_BBOX_X, boundingBox.ymin - self.TEXT_BBOX - self.INCREMENT_BBOX_Y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        return imageResized

    def imageCallback(self, img):
        self.imageQT = self.bridge.imgmsg_to_cv2(img, "bgr8")


class ObjectDetector:
    def __init__(self):
        rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknetCallback)
        self.onboardCamera = OnboardCamera("/onboard_camera/image_raw")
        self.boundingBoxes = []
        self.objectsDetected = []
        self.firstRedMask = rospy.get_param("/simParams/firstRedMask")
        self.secondRedMask = rospy.get_param("/simParams/secondRedMask")
        self.greenMask = rospy.get_param("/simParams/greenMask")
        self.TRAFFIC_LIGHT_PROB = rospy.get_param(
            "/simParams/TRAFFIC_LIGHT_PROB")
        self.STOP_SIGN_PROB = rospy.get_param("/simParams/STOP_SIGN_PROB")
        self.PERSON_PROB = rospy.get_param("/simParams/PERSON_PROB")

    def searchRed(self, hsvImage):
        firstMask = cv2.inRange(hsvImage, (self.firstRedMask[0], self.firstRedMask[1], self.firstRedMask[2]), (
            self.firstRedMask[3], self.firstRedMask[4], self.firstRedMask[5]))
        secondMask = cv2.inRange(hsvImage, (self.secondRedMask[0], self.secondRedMask[1], self.secondRedMask[2]), (
            self.secondRedMask[3], self.secondRedMask[4], self.secondRedMask[5]))
        maskRed = firstMask | secondMask

        if cv2.countNonZero(maskRed) > 0:
            return True

        return False

    def searchGreen(self, hsvImage):
        maskGreen = cv2.inRange(hsvImage, (self.greenMask[0], self.greenMask[1], self.greenMask[2]), (
            self.greenMask[3], self.greenMask[4], self.greenMask[5]))

        if cv2.countNonZero(maskGreen) > 0:
            return True

        return False

    def getObjects(self):
        return self.objectsDetected

    def update(self):
        return self.onboardCamera.update(self.getBoundingBoxes())

    def getBoundingBoxes(self):
        boundingBoxes = self.boundingBoxes

        return boundingBoxes

    def darknetCallback(self, msg):
        self.objectsDetected = []
        self.boundingBoxes = msg.bounding_boxes
        for boundingBox in self.boundingBoxes:
            if boundingBox.Class == "traffic light" and boundingBox.probability >= self.TRAFFIC_LIGHT_PROB:
                try:
                    croppedImage = self.onboardCamera.getCroppedImage(
                        self.getBoundingBoxes())
                    hsvImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
                    if self.searchRed(hsvImage):
                        rospy.loginfo("traffic light red")
                        boundingBox.Class = "traffic light red"
                    elif self.searchGreen(hsvImage):
                        boundingBox.Class = "traffic light green"
                        rospy.loginfo("traffic light green")
                except:
                    rospy.logerr("An exception occurred")
            elif boundingBox.Class == "stop sign" and boundingBox.probability < self.STOP_SIGN_PROB:
                pass
            elif boundingBox.Class == "person" and boundingBox.probability < self.PERSON_PROB:
                pass

            self.objectsDetected.append(boundingBox.Class)


class JetRacer:
    def __init__(self):
        super().__init__()
        self.TURN_VEL = rospy.get_param("/simParams/TURN_VEL")
        self.STRAIGHT_VEL = rospy.get_param("/simParams/STRAIGHT_VEL")

        self.frontRightVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/rear_right_wheel_velocity_controller/command', Float64, queue_size=10)
        self.frontLeftVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/rear_left_wheel_velocity_controller/command', Float64, queue_size=10)
        self.rearRightVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/front_right_wheel_velocity_controller/command', Float64, queue_size=10)
        self.rearLeftVelPublisher = rospy.Publisher(
            '/autonomous_vehicle/front_left_wheel_velocity_controller/command', Float64, queue_size=10)

    def forward(self):
        rightVelMsg = Float64()
        rightVelMsg.data = self.STRAIGHT_VEL
        leftVelMsg = Float64()
        leftVelMsg.data = -self.STRAIGHT_VEL
        self.frontRightVelPublisher.publish(rightVelMsg)
        self.frontLeftVelPublisher.publish(leftVelMsg)
        self.rearRightVelPublisher.publish(rightVelMsg)
        self.rearLeftVelPublisher.publish(leftVelMsg)

    def backward(self):
        rightVelMsg = Float64()
        rightVelMsg.data = -self.STRAIGHT_VEL
        leftVelMsg = Float64()
        leftVelMsg.data = self.STRAIGHT_VEL
        self.frontRightVelPublisher.publish(rightVelMsg)
        self.frontLeftVelPublisher.publish(leftVelMsg)
        self.rearRightVelPublisher.publish(rightVelMsg)
        self.rearLeftVelPublisher.publish(leftVelMsg)

    def right(self):
        rightVelMsg = Float64()
        rightVelMsg.data = -self.TURN_VEL
        self.frontRightVelPublisher.publish(rightVelMsg)
        self.frontLeftVelPublisher.publish(rightVelMsg)
        self.rearRightVelPublisher.publish(rightVelMsg)
        self.rearLeftVelPublisher.publish(rightVelMsg)

    def left(self):
        leftVelMsg = Float64()
        leftVelMsg.data = self.TURN_VEL
        self.frontRightVelPublisher.publish(leftVelMsg)
        self.frontLeftVelPublisher.publish(leftVelMsg)
        self.rearRightVelPublisher.publish(leftVelMsg)
        self.rearLeftVelPublisher.publish(leftVelMsg)

    def stop(self):
        stopVelMsg = Float64()
        stopVelMsg.data = 0
        self.frontRightVelPublisher.publish(stopVelMsg)
        self.frontLeftVelPublisher.publish(stopVelMsg)
        self.rearRightVelPublisher.publish(stopVelMsg)
        self.rearLeftVelPublisher.publish(stopVelMsg)


class LaneFollower:
    def __init__(self):
        rospy.loginfo("Loading model...")
        device = torch.device('cuda')
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(512, 2)
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(
            rospy.get_param("/simParams/modelPath")))
        self.model = self.model.cuda().eval().half()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        self.IMAGE_WIDTH = rospy.get_param("/simParams/IMAGE_WIDTH")
        self.IMAGE_HEIGHT = rospy.get_param("/simParams/IMAGE_HEIGHT")
        self.IMAGE_LANE_WIDTH = rospy.get_param("/simParams/IMAGE_LANE_WIDTH")
        self.IMAGE_LANE_HEIGHT = rospy.get_param(
            "/simParams/IMAGE_LANE_HEIGHT")

        rospy.loginfo("Model loaded")

    def preprocess(self, image):
        device = torch.device('cuda')
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def getLaneCenter(self, image):
        imageResized = cv2.resize(
            image, (self.IMAGE_LANE_WIDTH, self.IMAGE_LANE_HEIGHT))
        imageResolution = int(imageResized.shape[0])
        image = self.preprocess(imageResized).half()
        output = self.model(image).detach().cpu().numpy().flatten()

        steering = float(output[0])
        imagePixel = int(imageResolution * (steering / 2.0 + 0.5))
        prediction = cv2.circle(
            imageResized, (imagePixel, imagePixel), 8, (255, 0, 0), 3)
        image = cv2.resize(
            prediction, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

        return image


class AutonomousVehicle:
    def __init__(self, gridLayout, layoutOnboardCamera):
        super().__init__()
        self.STOP_TIME = rospy.get_param("/simParams/STOP_TIME")
        self.TIME_BETWEEN_STOP = rospy.get_param(
            "/simParams/TIME_BETWEEN_STOP")
        self.thirteenthState = rospy.get_param("/simParams/thirteenthState")
        self.twelfthState = rospy.get_param("/simParams/twelfthState")
        self.eleventhState = rospy.get_param("/simParams/eleventhState")
        self.tenthState = rospy.get_param("/simParams/tenthState")
        self.ninthState = rospy.get_param("/simParams/ninthState")
        self.eighthState = rospy.get_param("/simParams/eighthState")
        self.seventhState = rospy.get_param("/simParams/seventhState")
        self.sixthState = rospy.get_param("/simParams/sixthState")
        self.fifthState = rospy.get_param("/simParams/fifthState")
        self.fourthState = rospy.get_param("/simParams/fourthState")
        self.thirdState = rospy.get_param("/simParams/thirdState")
        self.secondState = rospy.get_param("/simParams/secondState")
        self.firstState = rospy.get_param("/simParams/firstState")

        self.robotName = rospy.get_param("/simParams/robotName")

        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.getStateRobot)

        self.state = 0

        self.gridLayout = gridLayout
        self.layoutOnboardCamera = layoutOnboardCamera

        self.timeStart = 0
        self.timeStop = 0
        self.timeRedStop = 0

        self.stopSignDetected = False
        self.stopSignTime = False
        self.redLightDetected = False

        self.objectDetector = ObjectDetector()
        self.objectDetector.update()
        self.laneFolower = LaneFollower()
        self.jetRacer = JetRacer()

    def setStartTime(self):
        self.timeStart = rospy.get_time()

    def updateCamera(self):
        image = self.objectDetector.update()
        image = self.laneFolower.getLaneCenter(image)
        imageWidget = QImage(
            image.data, image.shape[1], image.shape[0], image.shape[1] * image.shape[2], QImage.Format_BGR888)
        self.layoutOnboardCamera.setPixmap(QPixmap.fromImage(imageWidget))
        self.gridLayout.addWidget(self.layoutOnboardCamera, 3, 4)

    def getStateRobot(self, msg):
        for i in range(0, len(msg.name)):
            if msg.name[i] == self.robotName:
                self.robotOrientation = round(msg.pose[i].orientation.z, 2)
                self.robotOrientationW = round(msg.pose[i].orientation.w, 2)

    def execute(self):
        objectsDetected = self.objectDetector.getObjects()

        if "stop sign" in objectsDetected and not self.stopSignDetected and not self.stopSignTime:
            self.stopSignDetected = True
            self.stop()
            self.timeStop = time.time()
            rospy.loginfo("Stop due to stop sign")
        elif "traffic light red" in objectsDetected and not self.redLightDetected:
            self.redLightDetected = True
            self.stop()
            self.timeRedStop = time.time()
            rospy.loginfo("Stop due to traffic light red")
        elif "traffic light green" in objectsDetected and self.redLightDetected:
            self.redLightDetected = False
            self.deleteActor()
            rospy.loginfo("Actor deleted")
            rospy.loginfo("Green light detected, continue...")
            self.timeStart = self.timeStart + (time.time() - self.timeRedStop)
        elif self.stopSignDetected:
            if time.time() - self.timeStop > self.STOP_TIME:
                self.stopSignDetected = False
                self.stopSignTime = True
                self.timeStart = self.timeStart + (time.time() - self.timeStop)
        elif self.stopSignTime and time.time() - self.timeStop > self.TIME_BETWEEN_STOP:
            self.stopSignTime = False
            rospy.loginfo("10 seconds after stop sign detection")
        elif not self.stopSignDetected and not self.redLightDetected:
            if self.state == 13 and rospy.get_time() - self.timeStart > self.thirteenthState:
                self.stop()
                rospy.loginfo("Finished")
                time.sleep(2)
                sys.exit()
            elif self.state == 12 and rospy.get_time() - self.timeStart > self.twelfthState[0] and self.robotOrientationW <= self.twelfthState[1]:
                self.jetRacer.forward()
                self.state = 13
                rospy.loginfo("thirteenthState")
            elif self.state == 11 and rospy.get_time() - self.timeStart > self.eleventhState:
                self.jetRacer.right()
                self.state = 12
                rospy.loginfo("twelfthState")
            elif self.state == 10 and rospy.get_time() - self.timeStart > self.tenthState[0] and self.robotOrientation <= self.tenthState[1]:
                self.jetRacer.forward()
                self.state = 11
                rospy.loginfo("eleventhState")
            elif self.state == 9 and rospy.get_time() - self.timeStart > self.ninthState:
                self.jetRacer.right()
                self.state = 10
                rospy.loginfo("tenthState")
            elif self.state == 8 and rospy.get_time() - self.timeStart > self.eighthState[0] and self.robotOrientation < self.eighthState[1]:
                self.jetRacer.forward()
                self.state = 9
                rospy.loginfo("ninthState")
            elif self.state == 7 and rospy.get_time() - self.timeStart > self.seventhState:
                self.jetRacer.right()
                self.state = 8
                rospy.loginfo("eighthState")
            elif self.state == 6 and rospy.get_time() - self.timeStart > self.sixthState[0] and self.robotOrientation <= self.sixthState[1]:
                self.jetRacer.forward()
                self.state = 7
                rospy.loginfo("seventhState")
            elif self.state == 5 and rospy.get_time() - self.timeStart > self.fifthState:
                self.jetRacer.right()
                self.state = 6
                rospy.loginfo("sixthState")
            elif self.state == 4 and rospy.get_time() - self.timeStart > self.fourthState[0] and self.robotOrientation <= self.fourthState[1]:
                self.jetRacer.forward()
                self.state = 5
                rospy.loginfo("fifthState")
            elif self.state == 3 and rospy.get_time() - self.timeStart > self.thirdState:
                self.jetRacer.right()
                self.state = 4
                rospy.loginfo("fourthState")
            elif self.state == 2 and rospy.get_time() - self.timeStart > self.secondState[0] and self.robotOrientation < self.secondState[1]:
                self.jetRacer.forward()
                self.state = 3
                rospy.loginfo("thirdState")
            elif self.state == 1 and rospy.get_time() - self.timeStart > self.firstState:
                self.jetRacer.right()
                self.state = 2
                rospy.loginfo("secondState")
            elif self.state == 0 and rospy.get_time() - self.timeStart > 0:
                self.jetRacer.forward()
                rospy.loginfo("firstState")
                self.state = 1

    def forward(self):
        self.jetRacer.forward()

    def backward(self):
        self.jetRacer.backward()

    def right(self):
        self.jetRacer.right()

    def left(self):
        self.jetRacer.left()

    def stop(self):
        self.jetRacer.stop()


class AppUI(QMainWindow):
    def __init__(self):
        super().__init__()
        gridLayout = QGridLayout()
        centralWidget = QWidget()
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)
        layoutWidget = QWidget(centralWidget)
        layoutOnboardCamera = QLabel(layoutWidget)

        self.autonomousVehicle = AutonomousVehicle(
            gridLayout, layoutOnboardCamera)

        layoutMapCamera = QLabel(layoutWidget)
        self.mapCamera = MapCamera("/map_camera/image_raw",
                                   gridLayout, layoutMapCamera)
        self.mapCamera.update()
        self.autonomousVehicle.updateCamera()

        TITLE_WIDTH = rospy.get_param("/simParams/TITLE_WIDTH")
        BUTTON_WIDTH = rospy.get_param("/simParams/BUTTON_WIDTH")
        STOP_BUTTON_WIDTH = rospy.get_param("/simParams/STOP_BUTTON_WIDTH")
        STOP_BUTTON_HEIGHT = rospy.get_param("/simParams/STOP_BUTTON_HEIGHT")

        self.robotName = rospy.get_param("/simParams/robotName")
        self.START_POSE_X = rospy.get_param("/simParams/START_POSE_X")
        self.START_POSE_Y = rospy.get_param("/simParams/START_POSE_Y")
        self.START_POSE_Z = rospy.get_param("/simParams/START_POSE_Z")
        self.START_ORIENTATION_Z = rospy.get_param(
            "/simParams/START_ORIENTATION_Z")
        self.START_ORIENTATION_W = rospy.get_param(
            "/simParams/START_ORIENTATION_W")
        self.actorName = rospy.get_param("/simParams/actorName")
        self.DELETE_ACTOR_POSE_X = rospy.get_param(
            "/simParams/DELETE_ACTOR_POSE_X")
        self.DELETE_ACTOR_POSE_Y = rospy.get_param(
            "/simParams/DELETE_ACTOR_POSE_Y")
        self.DELETE_ACTOR_POSE_Z = rospy.get_param(
            "/simParams/DELETE_ACTOR_POSE_Z")

        simulatorControllerButton = QPushButton('Simulator Controller')
        simulatorControllerButton.setProperty('class', 'warning')
        simulatorControllerButton.setCheckable(True)
        simulatorControllerButton.setChecked(True)
        simulatorControllerButton.setFixedWidth(TITLE_WIDTH)
        gridLayout.addWidget(simulatorControllerButton, 0, 0, 3, 0,
                             alignment=Qt.AlignHCenter)

        forwardButton = QToolButton()
        forwardButton.setArrowType(Qt.UpArrow)
        forwardButton.setFixedWidth(BUTTON_WIDTH)
        forwardButton.clicked.connect(self.goForward)
        gridLayout.addWidget(forwardButton, 10, 2)
        backwardButton = QToolButton()
        backwardButton.setFixedWidth(BUTTON_WIDTH)
        backwardButton.setArrowType(Qt.DownArrow)
        backwardButton.clicked.connect(self.goBackward)
        gridLayout.addWidget(backwardButton, 12, 2)

        rightButton = QToolButton()
        rightButton.setFixedWidth(BUTTON_WIDTH)
        rightButton.setArrowType(Qt.RightArrow)
        rightButton.clicked.connect(self.goRight)
        gridLayout.addWidget(rightButton, 11, 3)

        leftButton = QToolButton()
        leftButton.setFixedWidth(BUTTON_WIDTH)
        leftButton.setArrowType(Qt.LeftArrow)
        leftButton.clicked.connect(self.goLeft)
        gridLayout.addWidget(leftButton, 11, 1)

        stopButton = QPushButton()
        stopButton.setProperty('class', 'danger')
        stopButton.setFixedWidth(STOP_BUTTON_WIDTH)
        stopButton.setFixedHeight(STOP_BUTTON_HEIGHT)
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
        self.actorPositionPublisher = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=10)

        self.simulationStarted = False

    def onPress(self):
        self.cameraTimer.start(1)
        self.cameraButton.setCheckable(False)

    def updateCameraTimer(self):
        self.autonomousVehicle.updateCamera()
        self.mapCamera.update()

    def deleteActor(self):
        deleteActorMsg = ModelState()
        deleteActorMsg.model_name = self.actorName
        deleteActorMsg.pose.position.x = self.DELETE_ACTOR_POSE_X
        deleteActorMsg.pose.position.y = self.DELETE_ACTOR_POSE_Y
        deleteActorMsg.pose.position.z = self.DELETE_ACTOR_POSE_Z
        self.actorPositionPublisher.publish(deleteActorMsg)

    def executeSimulation(self):
        self.autonomousVehicle.execute()

    def startSimulation(self):
        if not self.simulationStarted:
            self.autonomousVehicle.setStartTime()
            self.simulationTimer.timeout.connect(self.executeSimulation)
            self.simulationTimer.start(10)  # msec
            self.simulationStarted = True

    def restartSimulation(self):
        stateMsg = ModelState()
        stateMsg.model_name = self.robotName
        stateMsg.pose.position.x = self.START_POSE_X
        stateMsg.pose.position.y = self.START_POSE_Y
        stateMsg.pose.position.z = self.START_POSE_Z
        stateMsg.pose.orientation.z = self.START_ORIENTATION_Z
        stateMsg.pose.orientation.w = self.START_ORIENTATION_W
        self.actorPositionPublisher.publish(stateMsg)

    def goForward(self):
        self.autonomousVehicle.forward()

    def goBackward(self):
        self.autonomousVehicle.backward()

    def goRight(self):
        self.autonomousVehicle.right()

    def goLeft(self):
        self.autonomousVehicle.left()

    def stop(self):
        self.autonomousVehicle.stop()


rospy.init_node("simulator_controller")
app = QApplication([])
apply_stylesheet(app, theme='dark_blue.xml')
appUI = AppUI()
APP_MIN_WIDTH = rospy.get_param("/simParams/APP_MIN_WIDTH")
APP_MIN_HEIGHT = rospy.get_param("/simParams/APP_MIN_HEIGHT")
appUI.setMinimumSize(APP_MIN_WIDTH, APP_MIN_HEIGHT)
appUI.show()
sys.exit(app.exec_())
