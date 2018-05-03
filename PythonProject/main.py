import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import toolsInImageClass as toolsInImage
import PointPairsClass as ptsPairs
import poseEstimateClass as poseEstimate

cap = cv2.VideoCapture('greenBarsTest.avi')
frameCount = 0

while (cap.isOpened()):
    frameCount = frameCount + 1  #TODO: counter to count frame value
    ret, frame = cap.read()  #TODO: change varible ret to isFramePresent
    # hsvFrame = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    if frame is None:
        print(frameCount)

    leftImage = frame[0:540, 480:1440, :]  #TODO: keep variable for dimension specification
    rightImage = frame[540:1080, 480:1440, :]  #TODO define variable for dimension specification

    toolsObj = toolsInImage.InstrumentsInImage(frame, 2)
    leftImageTools, rightImageTools = toolsObj.getImageTools()

    corresPts = ptsPairs.PointPairsClass(leftImageTools, rightImageTools)

    leftImgCurves, rightImgCurves = corresPts.matchImgTools(leftImageTools, rightImageTools)

    leftImgMatchedPoints, rightImgMatchedPoints = corresPts.getPointPairs(leftImgCurves, rightImgCurves)

    for i in range(0, len(leftImgMatchedPoints)):
        for j in range(0, len(leftImgMatchedPoints[i])):
            for k in range(0, len(leftImgMatchedPoints[i][j])):
                cv2.circle(leftImage, (int(leftImgMatchedPoints[i][j][k, 0]), int(leftImgMatchedPoints[i][j][k, 1])), 3,
                           (0, 0, 255), -1)
    cv2.imshow('image', leftImage)

    poseEst = poseEstimate.poseEstimateClass(leftImgMatchedPoints, rightImgMatchedPoints)

    scenePts = poseEst.get3DPts(leftImgMatchedPoints, rightImgMatchedPoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
