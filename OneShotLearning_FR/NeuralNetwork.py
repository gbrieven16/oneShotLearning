import torch
from torch import nn
import torch.nn.functional as f

TYPE_ARCH = "1"


# ================================================================
#                    CLASS: Tripletnet
# ================================================================

class Tripletnet(nn.Module):

    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):

        embedded_x = self.embeddingnet([x])
        embedded_y = self.embeddingnet([y])
        embedded_z = self.embeddingnet([z])
        dist_a = f.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = f.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z


# ================================================================
#                    CLASS: Net
# ================================================================

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

        for i in range(len(data)):  # Siamese nets; sharing weights
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

        # ---- CASE 1: The triplet loss is used ----
        if len(data) == 1:
            return res[0]

        # ---- CASE 2: The cross entropy is used ----
        else:
            difference = torch.abs(res[1] - res[0]) # Computation of the difference of the 2 feature representations

            # TODO: Compute the avg of the difference
            # Return (1-avg_diff, avg_diff)
            #print("difference 1 is " + str(difference))
            difference = self.linear2(difference)
            #print("difference 2 is " + str(difference))
            return difference # Should be probability assign to each class? Why negative values