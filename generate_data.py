import numpy as np
import pandas as pd
import cv2 as CV


# Generating the training data
print('Generating Training data .... ')
np.random.seed(1)

pd.DataFrame(np.random.random_integers(30, 60, (1000, 4))).to_csv('./Dataset/ground-truth.csv', sep=',', header=False, index=False)
ground_truth = np.asarray(pd.read_csv('./Dataset/ground-truth.csv', sep=',', header=None))
imgsTrain = np.zeros((1000, 10000), dtype=np.uint8)

for i in range(len(imgsTrain)):
    img = np.reshape(imgsTrain[i], (100,100))
    CV.rectangle(img, (ground_truth[i][0], ground_truth[i][1]),
                 (ground_truth[i][0]+ground_truth[i][2], ground_truth[i][1]+ground_truth[i][3]),
                 (255,255,255), thickness=CV.FILLED
                 )
    imgsTrain[i] = img.flatten()

pd.DataFrame(imgsTrain).to_csv('./Dataset/trainingData.csv', sep=',', header=False, index=False)
print('Training data generated and saved.')
print('Generating test dataset ....')

# generating the test data
np.random.seed(123)

pd.DataFrame(np.random.random_integers(20, 70, (200, 4))).to_csv('./Dataset/ground-truth-test.csv', sep=',', header=False, index=False)
ground_truth_test = np.asarray(pd.read_csv('./Dataset/ground-truth-test.csv', sep=',', header=None))
imgsTest = np.zeros((200, 10000), dtype=np.uint8)

for i in range(len(imgsTest)):
    img = np.reshape(imgsTest[i], (100,100))
    CV.rectangle(img, (ground_truth_test[i][0], ground_truth_test[i][1]),
                 (ground_truth_test[i][0]+ground_truth_test[i][2], ground_truth_test[i][1]+ground_truth_test[i][3]),
                 (255,255,255), thickness=CV.FILLED
                 )
    imgsTest[i] = img.flatten()

pd.DataFrame(imgsTest).to_csv('./Dataset/testData.csv', sep=',', header=False, index=False)
print('Completed! Saved!')