import cv2 as cv

img = cv.imread('Media/Photos/park.jpg')

cv.imshow('Boston',img)

#Converting an image to graysacle
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

#Blur an image
blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
cv.imshow("Blur",blur)


# Edge Cascade

canny = cv.Canny(blur, 125, 175)
cv.imshow("Canny",canny)


# Dialating the image

dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow("Dialation",dilated)


# Eroding

eroded = cv.erode(dilated, (3,3), iterations=1)
cv.imshow("Eroded",eroded)

# Resize

resized = cv.resize(img,(500,500), interpolation=cv.INTER_AREA)
cv.imshow("Resized",resized)

# Cropping

cropped = img[50:200, 200:400]
cv.imshow("Cropped", cropped)


cv.waitKey(0)