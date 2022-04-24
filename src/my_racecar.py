from .racecar import Racecar
import traitlets
from adafruit_servokit import ServoKit
#from pynput import keyboard
import Jetson.GPIO as GPIO
import time

# Left motors
ENA = 33
IN1 = 21
IN2 = 22

# Right motors
ENB = 32
IN3 = 26
IN4 = 24

# 50 Hz
FREQUENCY = 50

class MyRacecar(Racecar):
	
	i2c_address = traitlets.Integer(default_value=0x40)
	steering_gain = traitlets.Float(default_value=-0.65)
	steering_offset = traitlets.Float(default_value=0)
	steering_channel = traitlets.Integer(default_value=0)
	throttle_gain = traitlets.Float(default_value=0.8)
	throttle_channel = traitlets.Integer(default_value=1)
	
	def __init__(self, *args, **kwargs):
		super(MyRacecar, self).__init__(*args, **kwargs)
		self.motorLeft = ENA
		self.forwardMotorLeft = IN1
		self.backwardMotorLeft = IN2
		self.motorRight = ENB
		self.forwardMotorRight = IN3
		self.backwardMotorRight = IN4
		self.FREQUENCY = FREQUENCY

		GPIO.setmode(GPIO.BOARD)
		GPIO.setwarnings(False)
		GPIO.setup(self.motorLeft, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.forwardMotorLeft, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.backwardMotorLeft, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.motorRight, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.forwardMotorRight, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.backwardMotorRight, GPIO.OUT, initial=GPIO.LOW)

		self.pwmLeft = GPIO.PWM(self.motorLeft, self.FRECUENCY)
		self.pwmRight = GPIO.PWM(self.motorRight, self.FRECUENCY)
		self.pwmLeft.start(0)
		self.pwmRight.start(0)
		self.steering_motor = 0
		self.throttle_motor = 0
	
	def _go_forward(self):
		GPIO.output(self.forwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.backwardMotorLeft, GPIO.LOW)
		GPIO.output(self.forwardMotorRight, GPIO.HIGH)
		GPIO.output(self.backwardMotorRight, GPIO.LOW)

	def _go_right(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.LOW)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.HIGH)

	def _go_left(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.LOW)

	def _stop(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.LOW)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.LOW)

	def _set_speed(self, percentageRight, percentageLeft):
		if percentageRight < 10:
			percentageRight = 10
		if percentageLeft < 10:
			percentageLeft = 10
		if percentageRight > 100:
			percentageRight = 100
		if percentageLeft > 100:
			percentageLeft = 100
		self.pwmLeft.start(percentageLeft)
		self.pwmRight.start(percentageRight)

	def __del__(self):
		self.stop()
		self.pwmLeft.stop()
		self.pwmRight.stop()
		GPIO.cleanup()

	@traitlets.observe('steering')
	def _on_steering(self, steering):
		# self.steering_motor.throttle = change['new'] * self.steering_gain + self.steering_offset
		if steering == 0.0:
			self._go_forward()
			self._set_speed(self.throttle_motor, self.throttle_motor)
		elif steering > 0.0:
			self._go_right()
			self._set_speed(steering, self.throttle_motor)
		elif steering < 0.0:
			self._go_right()
			self._set_speed(self.throttle_motor, steering)
		
		
		self.steering_motor = steering

	@traitlets.observe('throttle')
	def _on_throttle(self, throttle):
		self.throttle_motor = throttle
		# self.throttle_motor.throttle = change['new'] * self.throttle_gain