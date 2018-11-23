import sys
sys.path.append('../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv
from functions import overlapScore

testX = pd.read_csv('../Dataset/testData.csv', sep=',', header=None)
groundTruth = pd.read_csv('../Dataset/ground-truth-test.csv', sep=',', header=None)
predictedBB = pd.read_csv('test-result.csv', sep=',', header=None)

testX = np.asanyarray(testX, dtype=np.uint8)
groundTruth = np.asarray(groundTruth)
predictedBB = np.asarray(predictedBB, dtype=int)


for i in range(len(testX)):

    img = np.zeros((100,100,3))
    img[:,:, 0] = np.reshape(testX[i], (100, 100))

    cv.rectangle(img, (predictedBB[i][0], predictedBB[i][1]),
                     (predictedBB[i][0]+predictedBB[i][2], predictedBB[i][1]+predictedBB[i][3]), (0,255,0), 1)
    cv.rectangle(img, (groundTruth[i][0], groundTruth[i][1]),
                     (groundTruth[i][0]+groundTruth[i][2], groundTruth[i][1]+groundTruth[i][3]), (0,0,255), 1)
    filename = str(i) + '.jpg'
    cv.imwrite(filename, img)
