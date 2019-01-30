import torch
from torch import nn
import torch.nn.functional as f

"""
TYPE_ARCH 

1: without dropout, without batch normalization 
2: with dropout, without batch normalization  
3: without dropout, with batch normalization  
4: AlexNet architecture 
"""
TYPE_ARCH = "4AlexNet"  # "1default" "2def_drop" # "3def_bathNorm"
P_DROPOUT = 0.2 # Probability of each element to be dropped
WITH_NORM_BATCH = False
DIST_THRESHOLD = 0.02


# ================================================================
#                    CLASS: Tripletnet
# ================================================================

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, anchor, negative, positive):
        # --- Derivation of the feature representation ---
        embedded_anchor = self.embeddingnet([anchor])
        embedded_neg = self.embeddingnet([negative])
        embedded_pos = self.embeddingnet([positive])

        # --- Computation of the distance between them ---
        distance = f.pairwise_distance(embedded_anchor, embedded_neg, 2)  # (anchor - positive).pow(2).sum(1)  # .pow(.5)
        disturb = f.pairwise_distance(embedded_anchor, embedded_pos, 2)
        # losses = F.relu(distance_positive - distance_negative + self.margin)
        # return losses.mean() if size_average else losses.sum()
        return distance, disturb, embedded_anchor, embedded_neg, embedded_pos


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
        # loss = 0.5 *

        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * f.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


# ================================================================
#                    CLASS: Net
# Initial Basic Network that was trained
# ================================================================

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # ----------- For Feature Representation )-----------------
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(2304, 512)
        self.dropout = nn.Dropout(P_DROPOUT)

        # Last layer assigning a number to each class from the previous layer
        self.linear2 = nn.Linear(512, 2)

    def forward(self, data):
        res = []

        for i in range(len(data)):  # Siamese nets; sharing weights
            x = data[i]

            x = self.conv1(x)
            if WITH_NORM_BATCH: x = self.conv1_bn(x)
            x = f.relu(x)
            #x = self.dropout(x)
            x = self.pool1(x)

            x = self.conv2(x)
            if WITH_NORM_BATCH: x = self.conv2_bn(x)
            x = f.relu(x)
            #x = self.dropout(x)

            x = self.conv3(x)
            if WITH_NORM_BATCH: x = self.conv3_bn(x)
            x = f.relu(x)
            #x = self.dropout(x)

            x = x.view(x.shape[0], -1)  # To reshape
            x = self.linear1(x)
            res.append(f.relu(x))

        # ---- CASE 1: We just want the feature representation ----
        if len(data) == 1:
            return res[0]

        # ---- CASE 2: The cross entropy is used ----
        else:
            return self.get_final_pred(res[0], res[1])

    """ ----------------------- get_final_output ------------------------------------
        IN: feature_repr1: feature representation of the first face image
            feature_repr2: feature representation of the second face image 
            as_output: specifies to:
                return the final prediction if False  
                return the final values related to each class otherwise
    ------------------------------------------------------------------------------"""

    def get_final_output(self, feature_repr1, feature_repr2, as_output=True):
        # Computation of the difference of the 2 feature representations
        difference = torch.abs(feature_repr2 - feature_repr1)
        last_values = self.linear2(difference)
        # print("Last values are " + str(last_values))
        # print("Processed distance is " + str(self.avg_val_tensor(difference).requires_grad_(True)))
        # return self.avg_val_tensor(difference).requires_grad_(True) #
        if as_output:
            return last_values
        else:
            if not torch.cuda.is_available():
                return torch.squeeze(torch.argmax(last_values, dim=1)).cpu().item()
            else:
                return torch.squeeze(torch.argmax(last_values, dim=1)).cuda().item()


    """
    This function can be used to replace the last layer linear2 assigning a value to each class
    IN: A tensor ... x 16 
    OUT: A tensor 2 x 16 where the first value is avg_elem and the second is the 1-avg_elem
    
    Basic Idea: the higher the difference, the higher the weight to class 1 
    (if avg higher than the threshold => the second class (i.e class 1) has a higher assigned value)
    """

    def avg_val_tensor(self, init_tensor):
        new_content = []
        # Go through each batch
        for i, pred in enumerate(init_tensor):
            sum_el = 0
            for j, elem in enumerate(pred):
                sum_el += elem
            new_content.append([1, sum_el / (DIST_THRESHOLD*len(pred))])

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
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
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
            x = self.features(x)

            x = x.view(x.size(0), 16 * 4 * 4)
            res.append(self.linearization(x))

        # ---- CASE 1: We just want the feature representation ----
        if len(data) == 1:
            return res[0]

        # ---- CASE 2: The cross entropy is used ----
        else:
            difference = torch.abs(res[1] - res[0])  # Computation of the difference of the 2 feature representations
            return self.final_layer(difference)
