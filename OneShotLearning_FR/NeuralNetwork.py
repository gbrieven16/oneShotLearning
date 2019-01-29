import torch
from torch import nn
import torch.nn.functional as f

"""
TYPE_ARCH 

1: without dropout, without batch normalization 
2: with dropout, without batch normalization  
3: without dropout, with batch normalization  
"""
TYPE_ARCH = "4AlexNet"  # "1default" "2def_drop" # "3def_bathNorm"
WITH_DROPOUT = False
WITH_NORM_BATCH = False


# ================================================================
#                    CLASS: Tripletnet
# ================================================================

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, anchor, y, z):
        # --- Derivation of the feature representation ---
        embedded_anchor = self.embeddingnet([anchor])
        embedded_y = self.embeddingnet([y])
        embedded_z = self.embeddingnet([z])

        # --- Computation of the distance between them ---
        dist_a = f.pairwise_distance(embedded_anchor, embedded_y, 2) #(anchor - positive).pow(2).sum(1)  # .pow(.5)
        dist_b = f.pairwise_distance(embedded_anchor, embedded_z, 2)
        #losses = F.relu(distance_positive - distance_negative + self.margin)
        #return losses.mean() if size_average else losses.sum()
        return dist_a, dist_b, embedded_anchor, embedded_y, embedded_z

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        #loss = 0.5 *

        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * f.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


# ================================================================
#                    CLASS: Net
# ================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 7)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(2304, 512)
        self.linear2 = nn.Linear(512, 2)  # Last layer assigning a number to each class from the previous layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        res = []

        for i in range(len(data)):  # Siamese nets; sharing weights
            x = data[i]
            x = self.conv1(x)
            if WITH_NORM_BATCH: x = self.conv1_bn(x)
            x = f.relu(x)
            if WITH_DROPOUT: x = self.dropout(x)
            x = self.pool1(x)

            x = self.conv2(x)
            if WITH_NORM_BATCH: x = self.conv2_bn(x)
            x = f.relu(x)
            if WITH_DROPOUT: x = self.dropout(x)

            x = self.conv3(x)
            if WITH_NORM_BATCH: x = self.conv3_bn(x)
            x = f.relu(x)
            if WITH_DROPOUT: x = self.dropout(x)

            x = x.view(x.shape[0], -1)  # To reshape
            x = self.linear1(x)
            res.append(f.relu(x))

        # ---- CASE 1: We just want the feature representation ----
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


# ================================================================
#                    CLASS: DecoderNet
# ================================================================

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()

        if TYPE_ARCH != "4AlexNet":
            self.linear1 = nn.Linear(512, 2304)
        else:
            self.linear1 = nn.Linear(4096, 2304)

        self.conv3 = nn.ConvTranspose2d(9, 3, 13)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.ConvTranspose2d(64, 3, 7)
        self.sig = nn.Sigmoid()  # compress to a range (0, 1)

    def forward(self, data):
        x = self.linear1(data)
        x = x.view(x.size(0), 9, 16, 16)
        x = self.conv3(x)
        x = self.sig(x)

        return x


# ================================================================
#                    CLASS: AutoEncoder
# ================================================================
class AutoEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super(AutoEncoder, self).__init__()

        if TYPE_ARCH == "1default":
            self.encoder = Net().to(device)
        elif TYPE_ARCH == "4AlexNet":
            self.encoder = AlexNet().to(device)

        self.decoder = DecoderNet().to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# ================================================================
#                    CLASS: AlexNet
# ================================================================

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)



        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linearization = nn.Sequential(
            nn.Dropout(),
            nn.Linear(16 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.final_layer = nn.Linear(4096, num_classes)

    def forward(self, data):
        res = []

        for i in range(len(data)):  # Siamese nets; sharing weights
            x = data[i]
            #x = self.features(x)

            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool2(x)
            x = self.conv2(x)
            x = self.relu(x)
            #x = self.pool2(x)
            x = self.conv3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.relu(x)
            x = self.conv5(x)
            x = self.relu(x)
            x = self.pool2(x)

            x = x.view(x.size(0), 16 * 4 * 4)
            res.append(self.linearization(x))

        # ---- CASE 1: We just want the feature representation ----
        if len(data) == 1:
            return res[0]

        # ---- CASE 2: The cross entropy is used ----
        else:
            difference = torch.abs(res[1] - res[0])  # Computation of the difference of the 2 feature representations
            return self.final_layer(difference)

