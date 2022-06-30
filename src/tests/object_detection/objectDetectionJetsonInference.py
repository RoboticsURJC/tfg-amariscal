# Modified from https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py


import sys
import jetson.inference
import jetson.utils
import numpy as np
import cv2
import os
import time

import argparse
import sys


# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.",
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="",
                    nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="",
                    nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet",
                    help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0",
                    help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280,
                    help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720,
                    help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true',
                    default=(), help="run without display")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
font = jetson.utils.cudaFont()
import tensorflow as tf
from utils import visualization_utils as vis_util
from utils import label_map_util

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

# Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

start = time.time()

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
# frame_rate_calc = 1
# freq = cv2.getTickFrequency()
# font = cv2.FONT_HERSHEY_SIMPLEX


# Initialize Picamera and grab reference to the raw capture

# img, width, height = input.CaptureRGBA(zeroCopy=True)
while True:
    img = input.Capture()
    # cv2.imshow("CSI Camera", img)
    # This also acts as
    # t1 = cv2.getTickCount()

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = np.array(jetson.utils.cudaToNumpy(img, 1280, 720, 3))
    frame.setflags(write=1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)

    # Perform the actual detection by running the model with the image as input
    #(boxes, scores, classes, num) = sess.run(
    #    [detection_boxes, detection_scores,
    #     detection_classes, num_detections],
    #    feed_dict={image_tensor: frame_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     frame,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=8,
    #     min_score_thresh=0.40)

    # render the image
    output.Render(frame)
    #cv2.imshow(frame)

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break


# # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
# # i.e. a single-column array, where each item in the column has the pixel RGB value
# frame = cv2.imread("image.jpg")
# frame.setflags(write=1)
# frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# frame_expanded = np.expand_dims(frame_rgb, axis=0)

# # Perform the actual detection by running the model with the image as input
# (boxes, scores, classes, num) = sess.run(
#     [detection_boxes, detection_scores, detection_classes, num_detections],
#     feed_dict={image_tensor: frame_expanded})

# # Draw the results of the detection (aka 'visulaize the results')
# vis_util.visualize_boxes_and_labels_on_image_array(
#     frame,
#     np.squeeze(boxes),
#     np.squeeze(classes).astype(np.int32),
#     np.squeeze(scores),
#     category_index,
#     use_normalized_coordinates=True,
#     line_thickness=8,
#     min_score_thresh=0.40)

# # cv2.putText(frame, "FPS: {0:.2f}".format(
# #     frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

# # All the results have been drawn on the frame, so it's time to display it.
# cv2.imshow('Object detector', frame)
# filename = 'savedImage2.jpg'

# # Using cv2.imwrite() method
# # Saving the image
# cv2.imwrite(filename, frame)

# # print("Elapsed time: " + str(time.time() - start))

# # Press 'q' to quit
# cv2.waitKey()
