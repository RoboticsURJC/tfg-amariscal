# Modified from https://github.com/dusty-nv/jetson-inference/blob/master/python/examples/imagenet.py

import jetson.inference
import jetson.utils
import argparse
import sys

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
parser.add_argument("--width", type=int, default=640,
                    help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=480,
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

input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
font = jetson.utils.cudaFont()

while True:
    img = input.Capture()
    output.Render(img)

    if not input.IsStreaming() or not output.IsStreaming():
        break
