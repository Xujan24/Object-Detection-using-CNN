import torch.nn as nn
import torch.nn.functional as f
class cnn_model(nn.Module):

    def __init__(self):
        """
        The CNN consists of three convolutional layers, one fully connected layer and one output layer with 4 nodes.
        A kernel of size 5 with stride 1 is applied in each convolution layers. The first two convolutional layer are
        followed by a max pooling layer with kernel size 2 and stride 2.
        """
        super(cnn_model, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=0
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=5,
            stride=1,
            padding=0
        )


        self.fc1 = nn.Linear(
            in_features=18*18*128,
            out_features=2046
        )

        self.fc2 = nn.Linear(
            in_features=2046,
            out_features=4
        )

    def forward(self, val):
        val = f.relu(self.conv1(val))
        val = f.max_pool2d(val, kernel_size=2, stride=2)
        val = f.relu(self.conv2(val))
        val = f.max_pool2d(val, kernel_size=2, stride=2)
        val = f.relu(self.conv3(val))
        val = val.view(-1, 18*18*128)
        val = f.dropout(f.relu(self.fc1(val)), p=0.5, training=self.training)
        val = self.fc2(val)

        return val