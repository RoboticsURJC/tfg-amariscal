#!/usr/bin/env python3

from sensor_msgs.msg import Image as ImageCamera
from darknet_ros_msgs.msg import BoundingBoxes
import torchvision.transforms as transforms
from torch2trt import TRTModule
from std_msgs.msg import Int32
import Jetson.GPIO as GPIO
import numpy as np
import PIL.Image
import torch
import rospy
import time
import cv2


class JetRacer:
    def __init__(self, *args, **kwargs):
        super(JetRacer, self).__init__(*args, **kwargs)
        self.LEFT_MOTORS = rospy.get_param("/realParams/ENA")
        self.FRONT_LEFT_MOTOR = rospy.get_param("/realParams/IN1")
        self.REAR_LEFT_MOTOR = rospy.get_param("/realParams/IN2")
        self.RIGHT_MOTORS = rospy.get_param("/realParams/ENB")
        self.FRONT_RIGHT_MOTOR = rospy.get_param("/realParams/IN3")
        self.REAR_RIGHT_MOTOR = rospy.get_param("/realParams/IN4")
        self.FREQUENCY = rospy.get_param("/realParams/FREQUENCY")
        self.MIN_PWM_PERCENTAGE = rospy.get_param(
            "/realParams/MIN_PWM_PERCENTAGE")
        self.MAX_PWM_PERCENTAGE = rospy.get_param(
            "/realParams/MAX_PWM_PERCENTAGE")
        self.FORWARD_RANGE = rospy.get_param("/realParams/FORWARD_RANGE")
        self.STEERING_OFFSET = rospy.get_param("/realParams/STEERING_OFFSET")
        self.MIN_SPEED = rospy.get_param("/realParams/MIN_SPEED")
        self.TURNING_SPEED = rospy.get_param("/realParams/TURNING_SPEED")
        self.STEERING_GAIN = rospy.get_param("/realParams/STEERING_GAIN")

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.LEFT_MOTORS, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.FRONT_LEFT_MOTOR, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.REAR_LEFT_MOTOR, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.RIGHT_MOTORS, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.FRONT_RIGHT_MOTOR, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.REAR_RIGHT_MOTOR, GPIO.OUT, initial=GPIO.LOW)

        self.pwmLeft = GPIO.PWM(self.LEFT_MOTORS, self.FREQUENCY)
        self.pwmRight = GPIO.PWM(self.RIGHT_MOTORS, self.FREQUENCY)
        self.pwmLeft.start(0)
        self.pwmRight.start(0)
        self.throttleMotor = 0

    def goForward(self):
        GPIO.output(self.FRONT_LEFT_MOTOR, GPIO.LOW)
        GPIO.output(self.REAR_LEFT_MOTOR, GPIO.HIGH)
        GPIO.output(self.FRONT_RIGHT_MOTOR, GPIO.LOW)
        GPIO.output(self.REAR_RIGHT_MOTOR, GPIO.HIGH)

    def goRight(self):
        GPIO.output(self.FRONT_LEFT_MOTOR, GPIO.HIGH)
        GPIO.output(self.REAR_LEFT_MOTOR, GPIO.LOW)
        GPIO.output(self.FRONT_RIGHT_MOTOR, GPIO.LOW)
        GPIO.output(self.REAR_RIGHT_MOTOR, GPIO.HIGH)

    def goLeft(self):
        GPIO.output(self.FRONT_LEFT_MOTOR, GPIO.LOW)
        GPIO.output(self.REAR_LEFT_MOTOR, GPIO.HIGH)
        GPIO.output(self.FRONT_RIGHT_MOTOR, GPIO.HIGH)
        GPIO.output(self.REAR_RIGHT_MOTOR, GPIO.LOW)

    def stop(self):
        GPIO.output(self.FRONT_LEFT_MOTOR, GPIO.LOW)
        GPIO.output(self.REAR_LEFT_MOTOR, GPIO.LOW)
        GPIO.output(self.FRONT_RIGHT_MOTOR, GPIO.LOW)
        GPIO.output(self.REAR_RIGHT_MOTOR, GPIO.LOW)

    def setSpeed(self, percentageRight, percentageLeft):
        if percentageRight < self.MIN_PWM_PERCENTAGE:
            percentageRight = self.MIN_PWM_PERCENTAGE
        if percentageLeft < self.MIN_PWM_PERCENTAGE:
            percentageLeft = self.MIN_PWM_PERCENTAGE
        if percentageRight > self.MAX_PWM_PERCENTAGE:
            percentageRight = self.MAX_PWM_PERCENTAGE
        if percentageLeft > self.MAX_PWM_PERCENTAGE:
            percentageLeft = self.MAX_PWM_PERCENTAGE
        self.pwmLeft.start(percentageLeft)
        self.pwmRight.start(percentageRight)

    def __del__(self):
        self.stop()
        self.pwmLeft.stop()
        self.pwmRight.stop()
        GPIO.clseanup()

    def controller(self, output, throttle):
        self.throttleMotor = throttle
        if self.throttleMotor > 0:
            if round(output, 2) > self.FORWARD_RANGE:
                rightSpeed = self.STEERING_OFFSET + \
                    abs(output * self.STEERING_GAIN)
                if rightSpeed < self.MIN_SPEED:
                    rightSpeed = self.MIN_SPEED
                leftSpeed = self.TURNING_SPEED
                self.goRight()
                rospy.loginfo("Right: Original Steering:" + str(output) +
                              " result: " + str(rightSpeed))
            elif round(output, 2) < -self.FORWARD_RANGE:
                leftSpeed = self.STEERING_OFFSET + \
                    abs(output * self.STEERING_GAIN)
                if leftSpeed < self.MIN_SPEED:
                    leftSpeed = self.MIN_SPEED
                rightSpeed = self.TURNING_SPEED
                self.goLeft()
                rospy.loginfo("Left: Original Steering:" +
                              str(output) + " result: " + str(leftSpeed))
            else:
                self.goForward()
                rightSpeed = self.throttleMotor
                leftSpeed = self.throttleMotor
                rospy.loginfo("Forward: " + str(output))

            self.setSpeed(rightSpeed, leftSpeed)
        else:
            self.stop()
            self.setSpeed(0, 0)
            rospy.loginfo("Throttle is 0")


class ObjectDetector:
    def __init__(self):
        rospy.Subscriber(
            "/darknet_ros/bounding_boxes", BoundingBoxes, self.darknetCallback)
        rospy.Subscriber("/usb_cam/image_raw",
                         ImageCamera, self.cameraCallback)
        self.firstRedMask = rospy.get_param("/realParams/firstRedMask")
        self.secondRedMask = rospy.get_param("/realParams/secondRedMask")
        self.greenMask = rospy.get_param("/realParams/greenMask")
        self.PERSON_PROB = rospy.get_param("/realParams/PERSON_PROB")
        self.TRAFFIC_LIGHT_PROB = rospy.get_param(
            "/realParams/TRAFFIC_LIGHT_PROB")
        self.STOP_SIGN_PROB = rospy.get_param("/realParams/STOP_SIGN_PROB")
        self.STOP_TIME = rospy.get_param("/realParams/STOP_TIME")
        self.PERSON_WAIT_TIME = rospy.get_param("/realParams/PERSON_WAIT_TIME")
        self.TIME_BETWEEN_STOP = rospy.get_param(
            "/realParams/TIME_BETWEEN_STOP")
        self.MIN_STOP_AREA = rospy.get_param("/realParams/MIN_STOP_AREA")
        self.MIN_PERSON_AREA = rospy.get_param("/realParams/MIN_PERSON_AREA")
        self.objectsDetected = []
        self.image = None
        self.stopSignDetected = False
        self.stopSignTime = False
        self.timeStop = None
        self.timePerson = None
        self.redLightDetected = False
        self.personDetected = False

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

    def getObstacles(self):
        for objectClass in self.objectsDetected:
            rospy.loginfo("Objects: " + str(self.objectsDetected))
            if objectClass == "person":
                rospy.loginfo("Stop due to person")
                self.personDetected = True
                self.timePerson = time.time()
            elif objectClass == "traffic light red" and not self.redLightDetected:
                self.redLightDetected = True
                rospy.loginfo("Stop due to traffic light red")
            elif objectClass == "traffic light green" and self.redLightDetected:
                self.redLightDetected = False
                rospy.loginfo("Green light detected, continue...")
            elif objectClass == "stop sign" and not self.stopSignDetected and not self.stopSignTime:
                self.stopSignDetected = True
                self.timeStop = time.time()
                rospy.loginfo("Stop due to stop sign")

        if self.personDetected == True and not "person" in self.objectsDetected and time.time() - self.timePerson > self.PERSON_WAIT_TIME:
            self.personDetected = False

        if self.stopSignDetected:
            if time.time() - self.timeStop > self.STOP_TIME:
                self.stopSignDetected = False
                self.stopSignTime = True
                rospy.loginfo(
                    "5 seconds elapsed since stop sign was detected, continue")
        elif self.stopSignTime and time.time() - self.timeStop > self.TIME_BETWEEN_STOP:
            self.stopSignTime = False
            rospy.loginfo("10 seconds elapsed since stop sign was detected")

        if not self.stopSignDetected and not self.redLightDetected and not self.personDetected:
            return False
        else:
            rospy.loginfo("Stop: " + str(self.stopSignDetected) + " Red: " +
                          str(self.redLightDetected) + " Person: " + str(self.personDetected))
            return True

    def getImage(self):
        return self.image

    def cameraCallback(self, img):
        image = np.frombuffer(img.data, dtype=np.uint8).reshape(
            img.height, img.width, -1)
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def getCroppedImage(self, boundingBox):
        return self.image[boundingBox.ymin:boundingBox.ymax, boundingBox.xmin:boundingBox.xmax]

    def darknetCallback(self, msg):
        self.objectsDetected = []
        for boundingBox in msg.bounding_boxes:
            objectClass = boundingBox.Class
            if boundingBox.Class == "traffic light" and boundingBox.probability >= self.TRAFFIC_LIGHT_PROB:
                try:
                    croppedImage = self.getCroppedImage(boundingBox)
                    hsvImage = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
                    if self.searchRed(hsvImage):
                        objectClass = "traffic light red"
                        self.objectsDetected.append(objectClass)
                    elif self.searchGreen(hsvImage):
                        objectClass = "traffic light green"
                        self.objectsDetected.append(objectClass)
                except:
                    rospy.logerr("Could not get cropped image")
            elif boundingBox.Class == "stop sign" and boundingBox.probability >= self.STOP_SIGN_PROB and ((boundingBox.xmax - boundingBox.xmin) * (boundingBox.ymax - boundingBox.ymin)) > self.MIN_STOP_AREA:
                self.objectsDetected.append(objectClass)
            elif boundingBox.Class == "person" and boundingBox.probability >= self.PERSON_PROB and ((boundingBox.xmax - boundingBox.xmin) * (boundingBox.ymax - boundingBox.ymin)) > self.MIN_PERSON_AREA:
                self.objectsDetected.append(objectClass)
                rospy.loginfo("Size: " + str((boundingBox.xmax - boundingBox.xmin)
                                             * (boundingBox.ymax - boundingBox.ymin)))


class LaneFollower:
    def __init__(self):
        self.pub = rospy.Publisher("/center_image", Int32, queue_size=10)
        self.LANE_IMAGE_WIDTH = rospy.get_param("/realParams/LANE_IMAGE_WIDTH")
        self.LANE_IMAGE_HEIGHT = rospy.get_param(
            "/realParams/LANE_IMAGE_HEIGHT")
        rospy.loginfo("Loading model...")
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(
            '/home/jetson/github/tfg-amariscal/models/road_following_model_trt_3.pth'))
        rospy.loginfo("Model loaded")

    def preprocess(self, image):
        device = torch.device('cuda')
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def getLaneCenter(self, image):
        imageResized = cv2.resize(
            image, (self.LANE_IMAGE_WIDTH, self.LANE_IMAGE_HEIGHT))
        resolution = int(imageResized.shape[0])
        image = self.preprocess(imageResized).half()
        output = self.model(image).detach().cpu().numpy().flatten()
        centerImage = int(resolution * (float(output[0]) / 2.0 + 0.5))
        self.pub.publish(centerImage)

        return float(output[0])


class AutonomousVehicle:
    def __init__(self):
        super().__init__()
        self.THROTTLE = rospy.get_param("/realParams/THROTTLE")
        self.STOP_TIME = rospy.get_param("/realParams/STOP_TIME")
        self.TIME_BETWEEN_STOP = rospy.get_param(
            "/realParams/TIME_BETWEEN_STOP")
        self.objectDetector = ObjectDetector()
        self.laneFollower = LaneFollower()
        self.jetRacer = JetRacer()

    def execute(self):
        image = self.objectDetector.getImage()
        output = self.laneFollower.getLaneCenter(
            image)
        obstacles = self.objectDetector.getObstacles()
        if not obstacles:
            self.jetRacer.controller(output, self.THROTTLE)
        else:
            rospy.loginfo("Obstacle, stop")
            self.jetRacer.controller(output, 0)


rospy.init_node("autonomous_vehicle")
autonomousVehicle = AutonomousVehicle()
RATE = rospy.get_param("/realParams/RATE")
rate = rospy.Rate(RATE)

while not rospy.is_shutdown():
    autonomousVehicle.execute()
    rate.sleep()
