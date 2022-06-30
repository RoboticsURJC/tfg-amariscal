# Modified from https://github.com/NVIDIA-AI-IOT/jetracer

from jetcam.usb_camera import USBCamera
from torch2trt import TRTModule
from motorsJetson import Motors
from utils import preprocess
import Jetson.GPIO as GPIO
import numpy as np
import torch
import cv2


# Left motors
ENA = 33
IN1 = 21
IN2 = 22

# Right motors
ENB = 32
IN3 = 26
IN4 = 24

# 50 Hz
FREQUENCY = 50

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('../models/road_following_model_trt_3.pth'))

# from jetracer.nvidia_racecar import NvidiaRacecar
# car = NvidiaRacecar()

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

motors = Motors(ENA, IN1, IN2, ENB, IN3, IN4, FREQUENCY)


camera = USBCamera(width=224, height=224, capture_width=640,
				   capture_height=480, capture_device=0)

STEERING_GAIN = 0.5
STEERING_BIAS = 0.2

# car.throttle = 0.15

while True:
	image = camera.read()
	cv2.imshow("A", image)
	print("type: " + str(type(image)))
	image = preprocess(image).half()
	output = model_trt(image).detach().cpu().numpy().flatten()
	x = float(output[0])
	motors.w = x * STEERING_GAIN + STEERING_BIAS
	print("x: " + str(x) + " w: " + str(motors.w))
	motors.mySpeed()

	
	# car.steering = x * STEERING_GAIN + STEERING_BIAS
