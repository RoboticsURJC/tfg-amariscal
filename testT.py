import cv2
import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
# Apply image detector on a batch of image.
detector = hub.load(
    "https://tfhub.dev/google/imagenet/inception_v3/classification/4")
width = 1028
height = 1028

# Load image by Opencv2
img = cv2.imread('image_2.jpg')
# Resize to respect the input_shape
inp = cv2.resize(img, (width, height))

# # Convert img to RGB
rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

# # Is optional but i recommend (float convertion and convert img to tensor image)
rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

# # Add dims to rgb_tensor
rgb_tensor = tf.expand_dims(rgb_tensor, 0)

# # Now you can use rgb_tensor to predict label for exemple :
# plt.figure(figsize=(10, 10))

# boxes, scores, classes, num_detections = detector(rgb_tensor)
detector(rgb)

cv2.imshow('Tensorflow', rgb)
cv2.waitKey()
