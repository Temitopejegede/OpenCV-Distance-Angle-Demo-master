import cv2 as cv2
import numpy as np
import math

lower_green = np.array([40, 70, 80])
upper_green = np.array([70, 255, 255])

cap = cv2.VideoCapture(0)
# image = cv2.imread(cap)

while (cap.isOpened()):
    ret, frame = cap.read()
#Show the frames
    if ret == True:
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        thresh_image = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        threshed = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)[1]

        # contours = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = cv2.findContours(threshed, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)[-2]

        # contours = contours[0] if len(contours) == 2 else contours[1]

        

        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Car Video", frame)
#Press key 'q' to exit video
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    else:
        break
# ret, frame = cap.read()

# gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# thresh_image = cv2.threshold(
#     gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# contours = cv2.findContours(
#     thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]


# for i in contours:
#     x,y,w,h = cv2.boundingRect(i)
#     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

# cv2.imshow('thing', frame)
# cv2.waitKey(0)
cv2.destroyAllWindows()
