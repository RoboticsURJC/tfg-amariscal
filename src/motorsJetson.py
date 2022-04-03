from pynput import keyboard
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

class Motors:
	def __init__(self, ENA, IN1, IN2, ENB, IN3, IN4, FREQUENCY):
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

	def goForward(self):
		GPIO.output(self.forwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.backwardMotorLeft, GPIO.LOW)
		GPIO.output(self.forwardMotorRight, GPIO.HIGH)
		GPIO.output(self.backwardMotorRight, GPIO.LOW)

	def goBackward(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.HIGH)

	def goRight(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.LOW)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.HIGH)

	def goLeft(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.HIGH)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.LOW)

	def stop(self):
		GPIO.output(self.forwardMotorLeft, GPIO.LOW)
		GPIO.output(self.backwardMotorLeft, GPIO.LOW)
		GPIO.output(self.forwardMotorRight, GPIO.LOW)
		GPIO.output(self.backwardMotorRight, GPIO.LOW)

	def on_press(self, key):
		if key == keyboard.Key.esc:
			self.goBackward()
			self.setSpeed(29, 29)
			print('You Pressed Up Key!')
		elif key == keyboard.Key.f1:
			self.goForward()
			self.setSpeed(29, 29)
			print('You Pressed Down Key!')
		elif key == keyboard.Key.f2:
			self.goRight()
			self.setSpeed(40, 25)
			print('You Pressed Right Key!')
		elif key == keyboard.Key.f3:
			print('You Pressed Left Key!')
			self.goLeft()
			self.setSpeed(25, 40)
		elif key == keyboard.Key.f4:
			self.stop()
			print('You Pressed Stop Key!')

	def setSpeed(self, percentageRight, percentageLeft):
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


def main():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setwarnings(False)

	motors = Motors(ENA, IN1, IN2, ENB, IN3, IN4, FREQUENCY)
	# motors.goForward()
	# motors.goBackward()
	# motors.goRight()
	# motors.goLeft()
	speed = 29

	print("Motors running. Press CTRL+C to exit")
	while True:
		try:
			motors.setSpeed(speed, speed)
			with keyboard.Listener(
					on_press=motors.on_press) as listener:
							listener.join()
		finally:
			motors.__del__()

if __name__ == '__main__':
	main()

# sudo busybox devmem 0x700031fc 32 0x45
# sudo busybox devmem 0x6000d504 32 0x2
# sudo busybox devmem 0x70003248 32 0x46
# sudo busybox devmem 0x6000d100 32 0x00

# y = ax + b

# a is car.steering_gain
# b is car.steering_offset
# x is car.steering
