from .camera import Camera
import traitlets
import atexit
import cv2


class VideoCamera(Camera):
    capture_fps = traitlets.Integer(default_value=30)
    capture_width = traitlets.Integer(default_value=640)
    capture_height = traitlets.Integer(default_value=480)
    capture_device = traitlets.Integer(default_value=0)

    def __init__(self, *args, **kwargs):
        super(VideoCamera, self).__init__(*args, **kwargs)
        try:
            self.cap = cv2.VideoCapture('/home/alvaro/jetcam/output.mp4')

            re, image = self.cap.read()

            if not re:
                raise RuntimeError('Could not read image from camera.')

        except:
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.cap.release)

    def _gst_str(self):
        return 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, format=(string)BGR ! appsink'.format(self.capture_device, self.capture_width, self.capture_height, self.capture_fps)

    def _read(self):
        re, image = self.cap.read()
        if re:
            image_resized = cv2.resize(
                image, (int(self.width), int(self.height)))
            return image_resized
        else:
            raise RuntimeError('Could not read image from camera')
