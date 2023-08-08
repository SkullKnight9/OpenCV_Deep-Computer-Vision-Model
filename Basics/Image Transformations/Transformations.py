import cv2 as cv
import numpy as np

img = cv.imread('Media/Photos/park.jpg')

cv.imshow('Boston',img)

# Translations - shifting along x and y axis

def translate(img,x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x -> Shiffting left
# -y -> Shifting Up
# x -> Shifting Right
# y -> Shifting Down

translated = translate(img,100,-100)
cv.imshow("Translated",translated)

# Rotations
def rotate(img,angle,rotPoint = None):
    (height,width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img, rotMat, dimensions) 

rotated = rotate(img,-45)
cv.imshow("Rotated",rotated)

doublerotation = rotate(rotated,-45)
cv.imshow("2xRoated",doublerotation)


# Resizing
resized = cv.resize(img,(500,500), interpolation=cv.INTER_LINEAR)
cv.imshow("Resized",resized)


# Flipping
flip = cv.flip(img,1) # 0-> Vertically Flipping, 1-> Horizontal Flipping, -1-> Vertically and Horizontally flipping
cv.imshow("Flipped",flip)

#Cropping
cropped = img[200:400, 300:400]
cv.imshow("Cropped",cropped)

cv.waitKey(0)

