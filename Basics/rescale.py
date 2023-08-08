import cv2 as cv

img = cv.imread("Media/Photos/cat_large.jpg")
cv.imshow('img1',img)

def changeRes(width,height):
    #Live Videos only
    capture.set(3,width)
    capture.set(4,height)

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale) #frame.shape[1] width of img
    height = int(frame.shape[0] * scale) #frame.shape[0] height of img

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_image = rescaleFrame(img)
cv.imshow("Image", resized_image)

capture = cv.VideoCapture('Media/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    if not isTrue or frame is None:
        break
    
    frame_resized = rescaleFrame(frame, scale = .2)
    cv.imshow('Video', frame)
    cv.imshow('Video Resized',frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()
#cv.waitKey(0)