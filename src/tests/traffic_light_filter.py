import cv2

image = cv2.imread("../../resources/real_cropped_green.png")
hsvOriginal = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# hsv = cv2.resize(
#     croppedImage, (croppedImage.shape[1]*4, croppedImage.shape[0]*4))
alpha_ = 1.5  # Contrast control(1.0-3.0)
beta_ = 0  # Brightness control (0-100)
convert = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
hsvConvert = cv2.cvtColor(convert, cv2.COLOR_BGR2HSV)

maskGreenOriginal = cv2.inRange(hsvOriginal, (69, 199, 209), (133, 255, 255))
maskGreenConvert = cv2.inRange(hsvConvert, (69, 199, 209), (133, 255, 255))

cv2.imshow("Convert", convert)
cv2.imshow("Original", image)
cv2.imshow("Red Convert", maskGreenConvert)
cv2.imshow("Red Original", maskGreenOriginal)
cv2.waitKey(0)
