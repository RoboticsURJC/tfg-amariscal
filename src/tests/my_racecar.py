from .racecar import Racecar
import traitlets
#from adafruit_servokit import ServoKit
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

		self.pwmLeft = GPIO.PWM(self.motorLeft, self.FREQUENCY)
		self.pwmRight = GPIO.PWM(self.motorRight, self.FREQUENCY)
		self.pwmLeft.start(0)
		self.pwmRight.start(0)
		self.steering_motor = 0
		self.throttle_motor = 0
	
	def _go_forward(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.HIGH)

	def _go_right(self):
		GPIO.output(self.forwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.backwardMotorLeft, GPIO.LOW)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.HIGH)

	def _go_left(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.forwardMotorRight, GPIO.HIGH)
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
		self._stop()
		self.pwmLeft.stop()
		self.pwmRight.stop()
		GPIO.cleanup()

	def _on_steering(self, steering):
		# self.steering_motor.throttle = change['new'] * self.steering_gain + self.steering_offset
		# print("Forward: " + str(steering))
		
		if round(steering, 1) > 0.5:
			self._go_right()
			right_speed = 37 + abs(steering * 10)
			if right_speed < 38:
				right_speed = 40
			self._set_speed(right_speed, 28)
			#self._set_speed(40, self.throttle_motor + 2)
			print("Right: Original Steering:" + str(steering) + " result: " + str(right_speed) + " Throttle: " + str(23))
		elif round(steering, 1) < -0.5:
			self._go_left()
			left_speed = 37 + abs(steering * 10)
			if left_speed < 38:
				left_speed = 40
			self._set_speed(28, left_speed)
			#self._set_speed(self.throttle_motor + 2, 40)
			print("Left: Original Steering:" + str(steering) + " result: " + str(left_speed) + " Throttle: " + str(23))
		else:
			self._go_forward()
			self._set_speed(self.throttle_motor, self.throttle_motor)
			print("Forward: " + str(steering) + " Throttle: " + str(self.throttle_motor))
		
		#self.steering_motor = steering

	def _on_throttle(self, throttle):
		self.throttle_motor = throttle
		#rint("self.throttle_motor: " + str(throttle))
		# self.throttle_motor.throttle = change['new'] * self.throttle_gain
