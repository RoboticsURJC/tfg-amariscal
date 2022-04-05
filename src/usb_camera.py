from jetcam.usb_camera import USBCamera
import ipywidgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

camera = USBCamera(width=224, height=224, capture_width=640,
                   capture_height=480, capture_device=0)

image = camera.read()

print(image.shape)
print(camera.value.shape)

image_widget = ipywidgets.Image(format='jpeg')

image_widget.value = bgr8_to_jpeg(image)

display(image_widget)

camera.running = True


def update_image(change):
    image = change['new']
    image_widget.value = bgr8_to_jpeg(image)


camera.observe(update_image, names='value')
