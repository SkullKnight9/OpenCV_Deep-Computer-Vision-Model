import cv2 as cv

img = cv.imread("Media/Photos/cats.jpg")
cv.imshow("Cats",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("Gray",gray)

# Simple Thresholding

threshold, thresh = cv.threshold(gray,100,255, cv.THRESH_BINARY)
cv.imshow(" Simple Thresholded Image",thresh)

threshold, thresh_inv = cv.threshold(gray,100,255, cv.THRESH_BINARY_INV)
cv.imshow(" Simple  Inverted Thresholded Image",thresh_inv)

# Adaptive Thresholding

adpative_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow("Adaptive Thresholding",adpative_thresh)



cv.waitKey(0)