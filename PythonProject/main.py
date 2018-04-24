import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import sklearn.cluster
from skimage import measure


# from scipy.cluster.vq import vq, kmeans, whiten


def getCenterOrientation(M):
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']
    mu11 = M['m11'] / M['m00'] - cx * cy
    mu02 = M['m02'] / M['m00'] - cy ** 2
    mu20 = M['m20'] / M['m00'] - cx ** 2
    orientation = math.degrees(.5 * math.atan((2 * mu11) / (mu20 - mu02)))
    return np.array([cx, cy, orientation])


#def barCurvesandLines(toolContours, toolParams):


def findOrientation(toolParams,idx):
    x = toolParams[0:len(idx[0]), 0]
    y = toolParams[0:len(idx[0]), 1]
    z = np.polyfit(x, y, 1)

    orientation = z[0]
    return orientation


def getLinesandCurves(contour, centerLine):
    orientation = centerLine[2]
    center = np.mean(contour, axis=0)
    transformedData = transformContour(contour, center, orientation)
    tlIdx = 0
    trIdx = 0
    blIdx = 0
    brIdx = 0
    rightpartContour = []
    leftpartContour = []
    for i in range(0, len(contour)):
        if transformedData[0, i] > 0 and transformedData[1, i] > 0:
            rightpartContour.append(contour[i, 0, :])
            if brIdx == 0:
                brIdx = i
            if np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, brIdx]):
                brIdx = i
        if transformedData[0, i] < 0 and transformedData[1, i] > 0:
            leftpartContour.append(contour[i, 0, :])
            if blIdx == 0:
                blIdx = i
            if np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, blIdx]):
                blIdx = i
        if transformedData[0, i] < 0 and transformedData[1, i] < 0:
            leftpartContour.append(contour[i, 0, :])
            if tlIdx == 0:
                tlIdx = i
            if np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, tlIdx]):
                tlIdx = i
        if transformedData[0, i] > 0 and transformedData[1, i] < 0:
            rightpartContour.append(contour[i, 0, :])
            if trIdx == 0:
                trIdx = i
            if np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, trIdx]):
                trIdx = i
    #plt.scatter(transformedData[0, :], transformedData[1, :])
    corners = contour[[trIdx, tlIdx, blIdx, brIdx], 0, :]
    # plt.scatter(corners[:, 0], corners[:, 1])
    # plt.show()
    rightpartContour = np.asarray(rightpartContour)
    leftpartContour = np.asarray(leftpartContour)
    contourLine = []
    contourCurve = []
    rightlinePtsIdx, rightcurvePtsIdx = idxLineOrCurve(rightpartContour, corners)
    leftlinePtsIdx, leftcurvePtsIdx = idxLineOrCurve(leftpartContour, corners)

    contourCurve.append(rightpartContour[rightcurvePtsIdx])
    contourCurve.append(leftpartContour[leftcurvePtsIdx])

    contourLine.append(rightpartContour[rightlinePtsIdx])
    contourLine.append(leftpartContour[leftlinePtsIdx])

    return contourLine, contourCurve

def transformContour(contour, center, orientation):
    theta = -(math.atan(orientation))
    RMat = np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])

    transformedData = np.empty([2, len(contour)])
    for i in range(0, len(contour)):
        transformedData[:, i] = np.matmul(RMat, contour[i, 0, :] - center)

    return transformedData

def idxLineOrCurve(contour, corners):
    threshDist = 3
    isLinePt = np.zeros([1, len(contour)], dtype=bool)
    #linePts = []
    #curvePts = []
    for i in range(0, 2):
        point1 = corners[2 * i, :]
        point2 = corners[2 * i + 1, :]
        kLine = point2 - point1
        kLineNorm = kLine / np.linalg.norm(kLine)
        normVector = np.asarray([-kLineNorm[1], kLineNorm[0]])
        distance = abs((contour - point1).dot(normVector))
        isInlierLine = distance <= threshDist
        isLinePt = scipy.logical_or(isLinePt, isInlierLine)
    contour = np.asarray(contour)
    linePtsIdx = np.where(isLinePt == True)[1]
    curvePtsIdx = np.where(isLinePt == False)[1]
    return linePtsIdx, curvePtsIdx


def getLargestContours(imgContours):
    thresholdArea = 500
    imgContoursLargest = []
    #areaCon = []
    for i in range(0, len(imgContours)):
        area = cv2.contourArea(imgContours[i])
        if area > thresholdArea and hierarchy[0][i][2] < 0:
            imgContoursLargest.append(np.asarray(imgContours[i]))
            #areaCon.append(area)

    return imgContoursLargest


def groupTools(imgContours, numTools):
    orientMat = np.empty([len(imgContours), 3])
    for i in range(0, len(imgContours)):
        M = cv2.moments(imgContours[i])
        orientMat[i] = getCenterOrientation(M)

    kmeansLabels = sklearn.cluster.KMeans(n_clusters=numTools).fit(orientMat).labels_
    imgToolContours = []
    centerLine = []
    for i in range(0, numTools):
        idx = np.where(kmeansLabels == i)
        if len(idx[0]) >= 2:
            toolParams = orientMat[idx[0]]
            x = toolParams[0:len(idx[0]), 0]
            y = toolParams[0:len(idx[0]), 1]
            z = np.polyfit(x, y, 1)
            toolContours = []
            for i1 in range(0, len(idx[0])):
                toolContours.append(imgContours[idx[0][i1]])
         centerLine.append(np.array([toolParams[i][0], toolParams[i][1], z[0]]))
        imgToolContours.append(toolContours)
    centerLine = np.asarray(centerLine)

    return imgToolContours, centerLine


#def getToolCenterLine(toolParams):



def getToolCurves(imgContours):

    imgContoursLargest = getLargestContours(imgContours)

    imgToolContours, centerLine = groupTools(imgContoursLargest, 2)

    toolCurves =[]
    imgTools =[]
    for i in range(0, len(imgToolContours)):
        for j in range(0,len(imgToolContours[0])):
            contourLine, contourCurve = getLinesandCurves(imgToolContours[i][j], centerLine[i,:])
            toolCurves.append(contourCurve)
        imgTools.append(toolCurves)


    orientMat = np.empty([len(imgContoursLargest), 3])
    for i in range(0, len(imgContoursLargest)):
        M = cv2.moments(imgContoursLargest[i])
        orientMat[i] = getCenterOrientation(M)

    numTools = 2
    kmeansLabels = sklearn.cluster.KMeans(n_clusters=numTools).fit(orientMat).labels_

    imgTools = []
    for i in range(0, numTools):
        idx = np.where(kmeansLabels == i)
        if len(idx[0]) >= 2:
            toolParams = orientMat[idx[0]]
            toolContours = []
            for i1 in range(0, len(idx[0])):
                toolContours.append(imgContoursLargest[idx[0][i1]])
            x = toolParams[0:len(idx[0]), 0]
            y = toolParams[0:len(idx[0]), 1]
            z = np.polyfit(x, y, 1)
            toolCurves = []
            for j in range(0, len(toolContours)):
                contourLine, contourCurve = getLinesandCurves(toolContours[j], np.array([toolParams[j][0], toolParams[j][1], z[0]]))
                toolCurves.append(contourCurve)
                for j1 in range(0,len(contourCurve)):
                    plt.scatter(contourCurve[j1][:, 0], contourCurve[j1][:, 1])
        else:
            continue
        imgTools.append(toolCurves)
        plt.show()
    return imgTools


fundamentalMatrix = np.array([[-2.74739045956507e-09, -3.16582895519725e-08, 0.000657404101888771],
                              [9.34652517696475e-10, -7.67803377363370e-08, 0.00516819669563047],
                              [-0.000656279944127855, -0.00501173496143221, -0.0438609359119888]])

cap = cv2.VideoCapture('greenBarsTest.avi')

while cap.isOpened():
    ret, frame = cap.read()
    # hsvFrame = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)
    leftImage = frame[0:540, 480:1440, :]
    rightImage = frame[540:1080, 480:1440, :]

    leftImage = cv2.GaussianBlur(leftImage, (5, 5), 0)
    imageHSV = cv2.cvtColor(leftImage, cv2.COLOR_BGR2HSV)

    lowerThresGreen = np.array([25, 25, 0])
    upperThresGreen = np.array([130, 255, 255])

    imageMask = cv2.inRange(imageHSV, lowerThresGreen, upperThresGreen)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    imageMaskFiltered = cv2.morphologyEx(imageMask, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(imageMaskFiltered)
    # plt.show()

    img, imgContours, hierarchy = cv2.findContours(imageMaskFiltered, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    toolContours = getToolCurves(imgContours)


    # cv2.circle(leftImage, (imgContoursLargest[i][j, 0, 0], imgContoursLargest[i][j, 0, 1]), 3, (0, 0, 255), -1)
    # cv2.imshow('contours', leftImage)



    # toolCurves = []
    # for i in range(0, numTools):
    #     idx = np.where(kmeansLabels == i)
    #     if len(idx[0]) >= 2:
    #         toolParams = orientMat[idx[0]]
    #         toolContours = []
    #         for i1 in range(0, len(idx[0])):
    #             toolContours.append(imgContoursLargest[idx[0][i1]])

    #         x = toolParams[0:len(idx[0]), 0]
    #         y = toolParams[0:len(idx[0]), 1]
    #         z = np.polyfit(x, y, 1)
    #     for j in range(0, len(toolContours)):
    #         corners = getCornersContour(toolContours[j], np.array([toolParams[j][0], toolParams[j][1], z[0]]))
    #         for k in range(0, len(corners)):
    #             cv2.circle(leftImage, (corners[k, 0], corners[k, 1]), 3, (0, 255, 255), -1)
    #         linePts, curvePts = getLinesandCurves(toolContours[j], corners)
    #         toolCurves.append(curvePts)
    #         for k in range(0, len(curvePts)):
    #             cv2.circle(leftImage, (curvePts[k, 0], curvePts[k, 1]), 3, (0, 255, 0), -1)
    # cv2.imshow('contours', leftImage)

    # cv2.imshow('Mask', leftImage)
    # cv2.waitKey(40)
    # plt.scatter(corners[:, 0], corners[:, 1])
    # else:
    #       continue
