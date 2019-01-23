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
        # --- Derivation of the feature representation ---
        embedded_x = self.embeddingnet([x])
        embedded_y = self.embeddingnet([y])
        embedded_z = self.embeddingnet([z])

        # --- Computation of the distance between them ---
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
        self.linear2 = nn.Linear(512, 2)  # Last layer assigning a number to each class from the previous layer

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
            difference = torch.abs(res[1] - res[0])  # Computation of the difference of the 2 feature representations
            last_values = self.linear2(difference)
            # print("Last values are " + str(last_values))
            # print("Processed distance is " + str(self.avg_val_tensor(difference).requires_grad_(True)))
            # return self.avg_val_tensor(difference).requires_grad_(True) #
            return last_values

    """
    IN: A tensor ... x 16 
    OUT: A tensor 2 x 16 where the first value is avg_elem and the second is the 1-avg_elem
    """

    def avg_val_tensor(self, init_tensor):
        new_content = []
        for i, pred in enumerate(init_tensor):
            avg = 0
            for j, elem in enumerate(pred):
                avg += elem
            new_content.append([avg / len(pred), 1 - avg / len(pred)])

        return torch.tensor(new_content)  # , grad_fn=<AddBackward0>)  #requires_grad=True)
