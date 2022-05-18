import numpy as np
import rospy
import time
import cv2
import sys

from sensor_msgs.msg import Image as ImageCamera
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Camera:
    def __init__(self, topic, layout, layoutCamera, poseWidget):
        self.topic = topic
        self.image = np.zeros((3, 3, 3),
                              np.uint8)
        self.imageWidth = 320
        self.imageHeight = 240
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self.topic, ImageCamera, self.callback)
        self.layoutCamera = layoutCamera
        self.layout = layout
        self.poseWidget = poseWidget

    def update(self):  # Update camera
        imageResize = cv2.resize(
            self.image, (self.imageWidth, self.imageHeight))
        image = QImage(
            imageResize.data, imageResize.shape[1], imageResize.shape[0], imageResize.shape[1] * imageResize.shape[2], QImage.Format_RGB888)
        self.layoutCamera.setPixmap(QPixmap.fromImage(image))
        self.layout.addWidget(self.layoutCamera, 9, self.poseWidget)

    def callback(self, img):  # Callback camera
        self.image = self.bridge.imgmsg_to_cv2(img, "rgb8")


class AppUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.window = QWidget()
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel('Vehicle Controller'))
        self.forward = QPushButton('Forward')
        self.forward.clicked.connect(self.forward_clicked)
        self.layout.addWidget(self.forward)
        self.backward = QPushButton('Backward')
        self.backward.clicked.connect(self.backward_clicked)
        self.layout.addWidget(self.backward)
        self.stop = QPushButton('Stop')
        self.stop.clicked.connect(self.stop_clicked)
        self.layout.addWidget(self.stop)

        self.start = QPushButton('Start')
        self.layout.addWidget(self.start, 1, 1)
        # self.label = QLabel("First Axis")
        # self.label.setAlignment(Qt.AlignCenter)
        # self.layout.addWidget(self.label, 1, 1)
        # self.sliderFirstAxis = QSlider(Qt.Horizontal)
        # self.sliderFirstAxis.setMinimum(-5)
        # self.sliderFirstAxis.setMaximum(5)
        # self.sliderFirstAxis.setValue(0)
        # self.sliderFirstAxis.setTickPosition(QSlider.TicksBelow)
        # self.sliderFirstAxis.setTickInterval(1)
        # self.layout.addWidget(self.sliderFirstAxis, 1, 2)
        # self.sliderFirstAxis.valueChanged.connect(self.first_axis_slider)

        self.centralWidget = QWidget()
        self.centralWidget.setObjectName("centralwidget")
        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)
        self.layoutWidget = QWidget(self.centralWidget)
        self.layoutCamera = QLabel(self.layoutWidget)
        self.camera = Camera("/camera/image_raw",
                             self.layout, self.layoutCamera, 1)
        self.camera.update()

        rospy.init_node("controller")
        self.pushButton = QPushButton("Active Camera")
        self.layout.addWidget(self.pushButton, 8, 2)

        self.pushButton.setCheckable(True)
        self.pushButton.clicked.connect(self.on_press)

        self.pushButton.released.connect(self.on_release)
        self.pushButton.setCheckable(True)
        self.timer = QTimer()
        self.timer.timeout.connect(self.every_second_while_pressed)

        self.pubFrontRight = rospy.Publisher(
            '/rover/joint_wheel_back_right_velocity_controller/command', Float64, queue_size=10)
        self.pubFrontLeft = rospy.Publisher(
            '/rover/joint_wheel_back_left_velocity_controller/command', Float64, queue_size=10)
        self.pubBackRight = rospy.Publisher(
            '/rover/joint_wheel_front_right_velocity_controller/command', Float64, queue_size=10)
        self.pubBackLeft = rospy.Publisher(
            '/rover/joint_wheel_front_left_velocity_controller/command', Float64, queue_size=10)
        # self.pubFirstAxis = rospy.Publisher(
        #     '/rover/joint_first_axis_velocity_controller/command', Float64, queue_size=10)

        self.velocity = 0
        self.time = 0
        self.sub = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.callback_state)

    def callback_state(self, msg):  # Callback Gazebo Model States
        if self.time == 0:
            self.time = time.time()

    def on_release(self):
        self.timer.stop()

    def on_press(self):
        self.timer.start(1)

    # Update camera while camera button is pressed
    def every_second_while_pressed(self):
        self.camera.update()

    def forward_clicked(self):
        msg = Float64()
        msg.data = 5.0
        self.velocity = msg.data
        self.pubFrontRight.publish(msg)
        self.pubFrontLeft.publish(msg)
        self.pubBackRight.publish(msg)
        self.pubBackLeft.publish(msg)

    def backward_clicked(self):
        msg = Float64()
        msg.data = -3.0
        self.velocity = msg.data
        self.pubFrontRight.publish(msg)
        self.pubFrontLeft.publish(msg)
        self.pubBackRight.publish(msg)
        self.pubBackLeft.publish(msg)

    def stop_clicked(self):
        msg = Float64()
        msg.data = 0
        self.velocity = msg.data
        self.pubFrontRight.publish(msg)
        self.pubFrontLeft.publish(msg)
        self.pubBackRight.publish(msg)
        self.pubBackLeft.publish(msg)

    def first_axis_slider(self):
        msg = Float64()
        # msg.data = self.sliderFirstAxis.value() * 0.1


def darkMode(app):
    """Apply dark mode to Application"""
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(15, 15, 15))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(0, 87, 184).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)        # self.pubFirstAxis.publish(msg)


app = QApplication([])
darkMode(app)
myAPP = AppUI()
myAPP.show()
sys.exit(app.exec_())
