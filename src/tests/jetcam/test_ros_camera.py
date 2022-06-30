from jetcam.ros_camera import ROSCamera
import time
import cv2

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output.mp4', fourcc, 240, (320, 240))

camera = ROSCamera(width=320, height=240)
mytime = time.time()

while time.time() - mytime < 170:
    image = camera.read()
    if image is not None:
        try:
            colored_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            out.write(colored_frame)
            cv2.imshow('Frame', colored_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            raise RuntimeError('Could not show image')

out.release()
cv2.destroyAllWindows()
