import numpy as np
import pandas as pd

from functions import overlapScore

import torch

from cnn_model import cnn_model

testX = pd.read_csv('Dataset/testData.csv', sep=',', header=None)
groundTruth = pd.read_csv('Dataset/ground-truth-test.csv', sep=',', header=None)

testX = np.asanyarray(testX)
groundTruth = np.asarray(groundTruth)

model = cnn_model()
model.eval()
model.load_state_dict(torch.load('Model/cnn_model.pth'))


output = model(torch.Tensor(np.reshape(testX, (len(testX),1,100,100))))

output = output.detach().numpy()
output = output.astype(int)

score, _ = overlapScore(output, groundTruth)
score /= len(testX)
print('Test Average overlap score : %f' % score)

np.savetxt('Results/test-result.csv', output, delimiter=',')



