import cv2 as cv
import numpy as np

img = cv.imread("Media/Photos/cats.jpg")
cv.imshow("Cats",img)

# Averaging
average = cv.blur(img, (3,3))
cv.imshow("Avg Blur", average)

# Gaussian Blur
gauss = cv.GaussianBlur(img, (3,3), 0)
cv.imshow("Gaussian",gauss)

# Median Blur
median = cv.medianBlur(img,3)
cv.imshow("Median Blur",median)

# Bilateral
bilateral = cv.bilateralFilter(img,10,35,25)
cv.imshow("Bilateral",bilateral)

cv.waitKey(0)