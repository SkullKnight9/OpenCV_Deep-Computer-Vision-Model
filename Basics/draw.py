import cv2 as cv
import numpy as np
img = cv.imread('Media/photos/cat.jpg')
#cv.imshow("Cat",img)
#img[1000:1000, 1000:1000] = 255,0,0
#cv.imshow('Cat',img)

blank = np.zeros((500,500,3), dtype='uint8')
cv.imshow("Blank",blank)


#1. Paint Image a certain color
#blank[:] = 255,0,0
#cv.imshow('Blue',blank)


#blank[:] = 0,0,255
#cv.imshow('Red',blank)

#2. Paint a certain portion of the image a color
#blank[200:300, 300:400] = 0,255,0;
#cv.imshow("Box With Color",blank)

#3. Drawing a Rectangle
cv.rectangle(blank,(0,0),(250,500),(0,255,0), thickness=cv.FILLED)
cv.imshow("Rectangle",blank)

#4. Drawing a Circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1)
cv.imshow("Circle",blank)

#5. Drawing a Line
cv.line(blank, (100,250), (300,400), (255,255,255), thickness=3)
cv.imshow("Line",blank)


#6. Wrtiting text on an image
cv.putText(blank, 'Hello', (225,225), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
cv.imshow("Text",blank)




cv.waitKey(0)