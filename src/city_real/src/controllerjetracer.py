#!/usr/bin/env python3

from sensor_msgs.msg import Image as ImageCamera
from darknet_ros_msgs.msg import BoundingBoxes
import torchvision.transforms as transforms
from torch2trt import TRTModule
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
        ENA = 33
        IN1 = 21
        IN2 = 22
        ENB = 32
        IN3 = 26
        IN4 = 24
        FREQUENCY = 50
        self.motorLeft = ENA
        self.forwardMotorLeft = IN1
        self.backwardMotorLeft = IN2
        self.motorRight = ENB
        self.forwardMotorRight = IN3
        self.backwardMotorRight = IN4
        self.FREQUENCY = FREQUENCY

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.motorLeft, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.forwardMotorLeft, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.backwardMotorLeft, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.motorRight, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.forwardMotorRight, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.backwardMotorRight, GPIO.OUT, initial=GPIO.LOW)

        self.pwmLeft = GPIO.PWM(self.motorLeft, self.FREQUENCY)
        self.pwmRight = GPIO.PWM(self.motorRight, self.FREQUENCY)
        self.pwmLeft.start(0)
        self.pwmRight.start(0)
        self.throttleMotor = 0

    def goForward(self):
        GPIO.output(self.forwardMotorLeft, GPIO.LOW)
        GPIO.output(self.backwardMotorLeft, GPIO.HIGH)
        GPIO.output(self.forwardMotorRight, GPIO.LOW)
        GPIO.output(self.backwardMotorRight, GPIO.HIGH)

    def goRight(self):
        GPIO.output(self.forwardMotorLeft, GPIO.HIGH)
        GPIO.output(self.backwardMotorLeft, GPIO.LOW)
        GPIO.output(self.forwardMotorRight, GPIO.LOW)
        GPIO.output(self.backwardMotorRight, GPIO.HIGH)

    def goLeft(self):
        GPIO.output(self.forwardMotorLeft, GPIO.LOW)
        GPIO.output(self.backwardMotorLeft, GPIO.HIGH)
        GPIO.output(self.forwardMotorRight, GPIO.HIGH)
        GPIO.output(self.backwardMotorRight, GPIO.LOW)

    def stop(self):
        GPIO.output(self.forwardMotorLeft, GPIO.LOW)
        GPIO.output(self.backwardMotorLeft, GPIO.LOW)
        GPIO.output(self.forwardMotorRight, GPIO.LOW)
        GPIO.output(self.backwardMotorRight, GPIO.LOW)

    def setSpeed(self, percentageRight, percentageLeft):
        if percentageRight < 10:
            percentageRight = 10
        if percentageLeft < 10:
            percentageLeft = 10
        if percentageRight > 100:
            percentageRight = 100
        if percentageLeft > 100:
            percentageLeft = 100
        self.pwmLeft.start(percentageLeft)
        self.pwmRight.start(percentageRight)

    def __del__(self):
        self.stop()
        self.pwmLeft.stop()
        self.pwmRight.stop()
        GPIO.cleanup()

    def controller(self, output, throttle):
        self.throttleMotor = throttle
        if self.throttleMotor > 0:
            if round(output, 2) > 0.57:
                self.goRight()
                right_speed = 38 + abs(output * 10)
                if right_speed < 38:
                    right_speed = 43
                self.setSpeed(right_speed, 28)
                rospy.loginfo("Right: Original Steering:" + str(output) +
                              " result: " + str(right_speed))
            elif round(output, 2) < -0.57:
                self.goLeft()
                left_speed = 38 + abs(output * 10)
                if left_speed < 38:
                    left_speed = 43
                self.setSpeed(28, left_speed)
                rospy.loginfo("Left: Original Steering:" +
                              str(output) + " result: " + str(left_speed))
            else:
                self.goForward()
                self.setSpeed(self.throttleMotor, self.throttleMotor)
                rospy.loginfo("Forward: " + str(output))
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
        self.objectsDetected = []
        self.image = None
        self.stopSignDetected = False
        self.stopSignTime = False
        self.timeStop = None
        self.timePerson = None
        self.redLightDetected = False
        self.personDetected = False

    def searchRed(self, hsvImage):
        firstMask = cv2.inRange(hsvImage, (0, 170, 170), (10, 255, 255))
        secondMask = cv2.inRange(hsvImage, (170, 170, 170), (180, 255, 255))
        maskRed = firstMask | secondMask

        if cv2.countNonZero(maskRed) > 0:
            return True

        return False

    def searchGreen(self, hsvImage):
        maskGreen = cv2.inRange(hsvImage, (55, 199, 209), (133, 255, 255))

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

        if self.personDetected == True and not "person" in self.objectsDetected and time.time() - self.timePerson > 4:
            self.personDetected = False

        if self.stopSignDetected:
            if time.time() - self.timeStop > 5:
                self.stopSignDetected = False
                self.stopSignTime = True
                rospy.loginfo(
                    "5 seconds elapsed since stop sign was detected, continue")
        elif self.stopSignTime and time.time() - self.timeStop > 10:
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
            if boundingBox.Class == "traffic light" and boundingBox.probability >= 0.7:
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
            elif boundingBox.Class == "stop sign" and boundingBox.probability >= 0.9 and ((boundingBox.xmax - boundingBox.xmin) * (boundingBox.ymax - boundingBox.ymin)) > 16000:
                self.objectsDetected.append(objectClass)
            elif boundingBox.Class == "person" and boundingBox.probability >= 0.6 and ((boundingBox.xmax - boundingBox.xmin) * (boundingBox.ymax - boundingBox.ymin)) > 5000:
                self.objectsDetected.append(objectClass)
                # rospy.loginfo("Size: " + str((boundingBox.xmax - boundingBox.xmin)
                #                              * (boundingBox.ymax - boundingBox.ymin)))


class LaneFollower:
    def __init__(self):
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
        HEIGHT = 224
        WIDTH = 224
        imageResized = cv2.resize(image, (WIDTH, HEIGHT))
        resolution = int(imageResized.shape[0])
        image = self.preprocess(imageResized).half()
        output = self.model(image).detach().cpu().numpy().flatten()
        imagePixel = int(resolution * (float(output[0]) / 2.0 + 0.5))
        # Pub image Pixel

        return float(output[0])


class AutonomousVehicle:
    def __init__(self):
        super().__init__()
        self.stopTime = rospy.get_param("/realParams/stopTime")
        self.timeBetweenStop = rospy.get_param("/realParams/timeBetweenStop")
        self.objectDetector = ObjectDetector()
        self.laneFolower = LaneFollower()
        self.jetRacer = JetRacer()
        self.throttle = 27

    def execute(self):
        image = self.objectDetector.getImage()
        try:
            steering = self.laneFollower.getLaneCenter(
                image)
            obstacles = self.objectDetector.getObstacles()
            if not obstacles:
                self.jetRacer.controller(steering, self.throttle)
            else:
                rospy.loginfo("Obstacle, stop")
                self.jetRacer.controller(steering, 0)
            # prediction = cv2.circle(imageLowResolution, (imagePixel, imagePixel), 8, (255, 0, 0), 3)
        except:
            rospy.logerr('Could not get image')


rospy.init_node("controller")
autonomousVehicle = AutonomousVehicle()
rate = rospy.Rate(50)

while not rospy.is_shutdown():
    autonomousVehicle.execute()
    rate.sleep()
