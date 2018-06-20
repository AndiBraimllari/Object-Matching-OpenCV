import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
object = cv.imread('meh.png', 0)  # the aforementioned object 

width, height = object.shape[::-1]

while True:
    _, frame = capture.read()
    results = cv.matchTemplate(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), object, cv.TM_CCOEFF_NORMED)
    coords = np.where(results > 0.6)  # 0.6 is a chosen threshold
    a = 1
    for point in zip(*coords[::-1]):  #unpacks coords
        if a is 1:  # select only one
            cv.rectangle(frame, point, (point[0]+width, point[1]+height), (255, 69, 0), 2)
            a = 0

    cv.imshow('detect_me', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):  # waitKey returns a 32 bit, but we are interested only with first 8, thus 0xFF
        break

capture.release()
cv.destroyAllWindows()
