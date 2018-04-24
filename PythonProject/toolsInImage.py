import cv2
import numpy as np
import sklearn
import math

class InstrumentInImage(object):

    def __init__(self, inputStereoImage, numberOfTools = 2):
        self.inputRGBImage = inputStereoImage
        self.numberOfTools = numberOfTools

    def separateImages(self, inputStereoImage):
        leftImage = cv2.cvtColor(inputStereoImage[0:540, 480:1440, :], cv2.COLOR_BGR2HSV)
        rightImage = cv2.cvtColor(inputStereoImage[541:960, 480:1440, :], cv2.COLOR_BGR2HSV)
        return leftImage, rightImage

    def greenMask(self, inputRGBImage):
        imageHSV = cv2.cvtColor(inputRGBImage)
        lowerThresGreen = np.array([50, 25, 0])
        upperThresGreen = np.array([100, 255, 255])
        imageMask = cv2.inRange(imageHSV,lowerThresGreen,upperThresGreen)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        imageMaskFiltered = cv2.morphologyEx(imageMask, cv2.MORPH_CLOSE, kernel)
        return imageMaskFiltered

    def groupTools(self, inputBinImage):
        imgContours = cv2.findContours(inputBinImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thresholdArea = 100
        imgContoursLargest = []
        for cnt in imgContours[1]:
            area = cv2.contourArea(cnt)
            if area > thresholdArea:
                imgContoursLargest.append(cnt)

        orientMat = np.empty([len(imgContoursLargest), 3])
        for i in range(0, len(imgContoursLargest)):
            M = cv2.moments(imgContoursLargest[i])
            orientMat[i] = self.getCenterOrientation(M)

        kmeans = sklearn.cluster.KMeans(n_clusters=2).fit(orientMat).labels_






    def getCenterandOrientation(self, M):
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        mu11 = M['m11']/M['m00'] -cx*cy
        mu02 = M['m02']/M['m00'] - cy^2
        mu20 = M['m20']/M['m00'] - cx^2
        orientation = 0.5* np.atan((2*mu11)/(mu20-mu02))
        return orientation

    def getCornersContour(contour, centerLine):
        mLine = centerLine[2]
        originLine = centerLine[0:2]

        theta = -math.atan(mLine)
        RMat = np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])
        tlIdx = 0
        trIdx = 0
        blIdx = 0
        brIdx = 0
        transformedData = np.empty([2, len(contour)])
        for i in range(0, len(contour)):
            transformedData[:, i] = np.matmul(RMat, contour[i, 0, :]) - originLine
            if (transformedData[0, i] > 0 and transformedData[1, i] > 0):
                if brIdx == 0:
                    brIdx = i
                elif np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, brIdx]):
                    brIdx= i
            if (transformedData[0,i]  < 0 and transformedData[1,i] > 0):
                if blIdx == 0:
                    blIdx = i
                elif np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, blIdx]):
                    blIdx= i
            if (transformedData[0, i]  < 0 and transformedData[1, i] < 0):
                if tlIdx == 0:
                    tlIdx = i
                elif np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, tlIdx]):
                    tlIdx= i
            if (transformedData[0, i]  > 0 and transformedData[1, i] < 0):
                if trIdx == 0:
                    trIdx = i
                elif np.linalg.norm(transformedData[:, i]) >= np.linalg.norm(transformedData[:, trIdx]):
                    trIdx = i
        corners = contour[[trIdx, tlIdx, blIdx, brIdx], 0, :]
        return corners

    def getLinesandCurves(contour, corners):
        numPts = np.size(contour)
        threshDist = 2
        isLinePt = np.zeros([1, len(contour)], dtype=bool)
        linePts = []
        curvePts = []
        for i in range(0, 2):
            point1 = corners[2*i, :]
            point2 = corners[2*i + 1, :]
            kLine = point2-point1
            kLineNorm = kLine/np.linalg.norm(kLine)
            normVector = [-kLineNorm[1], kLineNorm[0]]
            distance = abs((contour[:, 0, :] - point1).dot(normVector))
            isInlierLine = distance <= threshDist
            isLinePt = scipy.logical_or(isLinePt, isInlierLine)

        inlierIdx = np.where(isLinePt == True)[1]
        linePts =contour[np.where(isLinePt == True)[1], 0, :]
        curvePts = contour[np.where(isLinePt == False)[1], 0, :]
        return linePts, curvePts