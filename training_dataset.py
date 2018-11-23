import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class training_dataset(Dataset):
    def __init__(self):
        trainX = np.asarray(pd.read_csv('./Dataset/trainingData.csv', sep=',', header=None))
        trainY = np.asarray(pd.read_csv('./Dataset/ground-truth.csv', sep=',', header=None))
        self.features_train = torch.Tensor(trainX)
        self.groundTruth_train = torch.Tensor(trainY)

        self.len = len(trainX)

        # testX = np.asarray(pd.read_csv('./Dataset/testData.csv', sep=',', header=None))
        # testY = np.asarray(pd.read_csv('./Dataset/ground-truth-test.csv', sep=',', header=None))
        # self.features_test = torch.Tensor(testX)
        # self.groundTruth_test = torch.Tensor(testY)

    def __getitem__(self, item):
        return self.features_train[item], self.groundTruth_train[item]

    def __len__(self):
        return self.len