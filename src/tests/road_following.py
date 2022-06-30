# Modified from https://github.com/NVIDIA-AI-IOT/jetracer

import torchvision.transforms as transforms
from jetcam.usb_camera import USBCamera
from torch2trt import TRTModule
from motorsJetson import Motors
import Jetson.GPIO as GPIO
import PIL.Image
import torch
import cv2

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    device = torch.device('cuda')
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


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
model_trt.load_state_dict(torch.load(
    '../models/road_following_model_trt.pth'))

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

motors = Motors(ENA, IN1, IN2, ENB, IN3, IN4, FREQUENCY)

camera = USBCamera(width=224, height=224, capture_width=640,
                   capture_height=480, capture_device=0)

STEERING_GAIN = 0.5
STEERING_BIAS = 0.2

while True:
    image = camera.read()
    cv2.imshow("Image", image)
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = float(output[0])
    motors.w = x * STEERING_GAIN + STEERING_BIAS
    print("Output: " + str(x) + " Steering: " + str(motors.w))
    motors.mySpeed()
