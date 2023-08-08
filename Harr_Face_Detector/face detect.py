import cv2 as cv

# img = cv.imread("Media/Photos/fag2.jpg")
# cv.imshow("Person",img)

# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("Gray",gray)


capture = cv.VideoCapture('Media/Videos/fag3.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

haar_cascade = cv.CascadeClassifier('D:\OpenCV\Harr_Face_Detector\haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)


print(f'Number of faces found = {len(faces_rect)}')


for (x,y,w,h) in faces_rect:
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected Faces',frame)
cv.waitKey(0)