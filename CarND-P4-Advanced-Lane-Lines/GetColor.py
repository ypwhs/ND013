import cv2
import numpy as np
frame = cv2.imread('test.jpg')


def click(event, x, y, flags, param):
    if event == 1:
        color = np.uint8([[frame[y, x]]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        print(hsv_color)

cv2.namedWindow('raw')
cv2.setMouseCallback('raw', click)

while True:
    cv2.imshow('raw', frame)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
