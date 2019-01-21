import torch
from torch import nn
import torch.nn.functional as f


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 7)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.linear1 = nn.Linear(2304, 512)

        self.linear2 = nn.Linear(512, 2)

    def forward(self, data):
        res = []
        for i in range(2):  # Siamese nets; sharing weights
            x = data[i]
            x = self.conv1(x)
            x = f.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = f.relu(x)
            x = self.conv3(x)
            x = f.relu(x)

            x = x.view(x.shape[0], -1)
            x = self.linear1(x)
            res.append(f.relu(x))

        res = torch.abs(res[1] - res[0])
        res = self.linear2(res)
        return res