import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # here '0' is the id number
detector = HandDetector(maxHands=1)  # because we just want the single hand for our data collection part
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


offset = 20  # here offset is only for proving right space
imgSize = 300

folder = "Data/C"
counter = 0

labels=["A", "B","C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction,index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)


        else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal+hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite,draw=False)
        cv2.rectangle(imgOutput, (x - offset, y - offset-50), (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255), 2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)  # this will just a one millisecond delay


# webcam is ready and the second part would be to crop the image once we get the proper img.
