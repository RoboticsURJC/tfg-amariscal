from sensor_msgs.msg import Image as ImageCamera
from cv_bridge import CvBridge
from .camera import Camera
import numpy as np
import threading
import traitlets
import atexit
import rospy
import cv2


class ROSCamera(Camera):

    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=320)
    capture_height = traitlets.Integer(default_value=240)
    capture_device = traitlets.Integer(default_value=0)

    def __init__(self, *args, **kwargs):
        super(ROSCamera, self).__init__(*args, **kwargs)
        rospy.init_node("ros_camera_reader")
        self.image = None
        self.bridge = CvBridge()
        try:
            self.sub = rospy.Subscriber(
                "/camera/image_raw", ImageCamera, self._callback)
            re = True

            if not re:
                print("'Could not read image from camera.'")

        except:
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

    def _callback(self, img):  # Callback camera
        self.image = np.frombuffer(img.data, dtype=np.uint8).reshape(
            img.height, img.width, -1)

    def _read(self):
        try:
            if self.image is not None:
                image_resized = cv2.resize(
                    self.image, (int(self.width), int(self.height)))
                return image_resized
        except:
            raise RuntimeError('Could not read image from camera')
