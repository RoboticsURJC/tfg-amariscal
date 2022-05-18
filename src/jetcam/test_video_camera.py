from multiprocessing.connection import wait
from cv2 import waitKey
from jetcam.video_camera import VideoCamera
import cv2
import numpy as np
from PIL import Image
import time
from matplotlib import pyplot as plt

camera = VideoCamera(width=320, height=240)
mytime = time.time()

while time.time() - mytime < 20:
    image = camera.read()
    try:
        cv2.imshow('VideoCamera', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        raise RuntimeError('Could not show image')

cv2.destroyAllWindows()
print("Finished reading from VideoCamera")
