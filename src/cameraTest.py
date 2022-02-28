import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

def read_cam():
     cap = cv2.VideoCapture(0)
     if cap.isOpened():
         cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
         while True:
             ret_val, img = cap.read();
             cv2.imshow('demo',img)
             if cv2.waitKey(1) == ord('q'):
                  break
     else:
         print ("camera open failed")

     cv2.destroyAllWindows()


if __name__ == '__main__':
     print(cv2.getBuildInformation())
     Gst.debug_set_active(True)
     Gst.debug_set_default_threshold(0)
     read_cam()
