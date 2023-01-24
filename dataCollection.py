import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # here '0' is the id number
detector = HandDetector(maxHands=1)  # because we just want the single hand for our data collection part


offset = 20  # here offset is only for proving right space
imgSize = 300

folder = "Data/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]  # only one hand that's why i'm using the '0'
        x, y, w, h = hand['bbox']  # get the bounding box information

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]  # it is basically a matrix you have to define the starting height & ending height|| starting width & ending width

        # overlay of the img top of the white img
        imgCropShape = imgCrop.shape

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize

        else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal+hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", img)

    key = cv2.waitKey(1)  # this will just a one millisecond delay
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

# webcam is ready and the second part would be to crop the image once we get the proper img.

