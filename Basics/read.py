import cv2 as cv

img = cv.imread('Media/Photos/cat.jpg')
cv.imshow('img1',img)
cv.waitKey(0)

capture = cv.VideoCapture('Media/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
#cv.waitKey(0)