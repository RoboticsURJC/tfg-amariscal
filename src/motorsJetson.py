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

FREQUENCY = 50


def main():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENB, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

    pwmLeft = GPIO.PWM(ENA, FREQUENCY)
    pwmRight = GPIO.PWM(ENB, FREQUENCY)
    percentage = 20
    pwmLeft.start(percentage)
    pwmRight.start(percentage)

    print("PWM running. Press CTRL+C to exit")
    try:
        while True:
            # time.sleep(0.25)
            a = 0
            # print("Running")
            # pwmLeft.ChangeDutyCycle(percentage)
            # pwmRight.ChangeDutyCycle(percentage)
    finally:
        pwmLeft.stop()
        pwmRight.stop()
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)
        GPIO.cleanup()


if __name__ == '__main__':
    main()

# sudo busybox devmem 0x700031fc 32 0x45
# sudo busybox devmem 0x6000d504 32 0x2
# sudo busybox devmem 0x70003248 32 0x46
# sudo busybox devmem 0x6000d100 32 0x00
