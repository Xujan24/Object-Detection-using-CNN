import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from functions import overlapScore


from cnn_model import *
from training_dataset import *

def train_model(net, dataloader, batchSize, lr_rate, momentum):
    criterion = nn.MSELoss()
    optimization = optim.SGD(net.parameters(), lr=lr_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimization, step_size=30, gamma=0.1)

    for epoch in range(50):

        scheduler.step()

        for i, data in enumerate(dataloader):
            optimization.zero_grad()

            inputs, labels = data

            inputs, labels = inputs.view(batchSize,1, 100, 100), labels.view(batchSize, 4)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimization.step()

            pbox = outputs.detach().numpy()
            gbox = labels.detach().numpy()
            score, _ = overlapScore(pbox, gbox)

            print('[epoch %5d, step: %d, loss: %f, Average Score = %f' % (epoch+1, i+1, loss.item(), score/batchSize))

    print('Finish Training')


if __name__ == '__main__':
    # Hyper parameters
    learning_rate = 0.000001
    momentum = 0.9
    batch = 100
    no_of_workers = 2
    shuffle = True


    trainingdataset = training_dataset()
    dataLoader = DataLoader(
        dataset=trainingdataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=no_of_workers
    )

    model = cnn_model()
    model.train()

    train_model(model, dataLoader, batch,learning_rate, momentum)
    torch.save(model.state_dict(), './Model/cnn_model.pth')


