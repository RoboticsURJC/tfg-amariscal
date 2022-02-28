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
    def __init__(self, ENA, IN1, IN2, ENB, IN3, IN4, FRECUENCY):
        self.motorLeft = ENA
        self.forwardMotorLeft = IN1
        self.backwardMotorLeft = IN2
        self.motorRight = ENB
        self.forwardMotorRight = IN3
        self.backwardMotorRight = IN4
        self.FRECUENCY = FRECUENCY

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
        GPIO.output(self.forwardMotorLeft, GPIO.HIGH)
        GPIO.output(self.backwardMotorLeft, GPIO.LOW)
        GPIO.output(self.forwardMotorRight, GPIO.LOW)
        GPIO.output(self.backwardMotorRight, GPIO.LOW)

    def goLeft(self):
        GPIO.output(self.forwardMotorLeft, GPIO.LOW)
        GPIO.output(self.backwardMotorLeft, GPIO.LOW)
        GPIO.output(self.forwardMotorRight, GPIO.HIGH)
        GPIO.output(self.backwardMotorRight, GPIO.LOW)

    def stop(self):
        GPIO.output(self.forwardMotorLeft, GPIO.LOW)
        GPIO.output(self.backwardMotorLeft, GPIO.LOW)
        GPIO.output(self.forwardMotorRight, GPIO.LOW)
        GPIO.output(self.backwardMotorRight, GPIO.LOW)

    def setSpeed(self, percentage):
        if percentage < 20:
            percentage = 20
        elif percentage > 100:
            percentage = 100
        self.pwmLeft.start(percentage)
        self.pwmRight.start(percentage)
        print("Motors running. Press CTRL+C to exit")
        try:
            while True:
                # time.sleep(0.25)
                a = 0
                # print("Running")
                # pwmLeft.ChangeDutyCycle(percentage)
                # pwmRight.ChangeDutyCycle(percentage)
        finally:
            self.__del__()

    def __del__(self):
        self.stop()
        self.pwmLeft.stop()
        self.pwmRight.stop()
        GPIO.cleanup()


def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    motors = Motors(ENA, IN1, IN2, ENB, IN3, IN4, FREQUENCY)
    motors.goForward()
    # motors.goBackward()
    # motors.goRight()
    # motors.goLeft()
    motors.setSpeed(20)


if __name__ == '__main__':
    main()

# sudo busybox devmem 0x700031fc 32 0x45
# sudo busybox devmem 0x6000d504 32 0x2
# sudo busybox devmem 0x70003248 32 0x46
# sudo busybox devmem 0x6000d100 32 0x00
