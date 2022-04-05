# Modified from https://github.com/NVIDIA-AI-IOT/jetracer

from jetcam.usb_camera import USBCamera
from torch2trt import TRTModule
from utils import preprocess
import numpy as np
import torch
import cv2

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('road_following_model_trt.pth'))

# from jetracer.nvidia_racecar import NvidiaRacecar

# car = NvidiaRacecar()

camera = USBCamera(width=224, height=224, capture_width=640,
                   capture_height=480, capture_device=0)


STEERING_GAIN = 0.75
STEERING_BIAS = 0.00

# car.throttle = 0.15

while True:
    image = camera.read()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = float(output[0])
    print(x)
    # cv2.imshow('A', image)
    # car.steering = x * STEERING_GAIN + STEERING_BIAS
