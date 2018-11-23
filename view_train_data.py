import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridSpec

train_x = np.asarray(pd.read_csv('./Dataset/trainingData.csv', sep=',', header=0))

fig = plt.figure()
grid = gridSpec.GridSpec(ncols=4, nrows=1)

fig.add_subplot(grid[0,0])
plt.imshow(np.transpose(np.reshape(train_x[15], (100,100))))
fig.add_subplot(grid[0,1])
plt.imshow(np.transpose(np.reshape(train_x[75], (100,100))))
fig.add_subplot(grid[0,2])
plt.imshow(np.transpose(np.reshape(train_x[125], (100,100))))
fig.add_subplot(grid[0,3])
plt.imshow(np.transpose(np.reshape(train_x[189], (100,100))))

plt.show()