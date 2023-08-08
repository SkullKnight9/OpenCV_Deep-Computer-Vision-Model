import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Media/Photos/park.jpg')
cv.imshow("Boston",img)

# plt.imshow(img)
# plt.show()

#BGR To Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #grayscale version, use to see pixel intensity
cv.imshow("Gray",gray)

#BGR to HSV
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow("HSV",hsv)


#BGR to LAB
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow("LAB",lab)

#BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow("RGB",rgb)

#plt.imshow(rgb)
#plt.show()

#HSV to BGR

hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow("HSV_BGR",hsv_bgr)



# no direct conversions for lets say, Grayscale to HSV, you have to convert it BGR first !
cv.waitKey(0)