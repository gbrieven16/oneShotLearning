import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import matplotlib

from Dataprocessing import CENTER_CROP

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
#                   GLOBAL VARIABLES
# ================================================================

"""
TYPE_ARCH (related to the embedding Network)

1: without dropout, without batch normalization 
2: with dropout, without batch normalization  
3: without dropout, with batch normalization  
4: AlexNet architecture 
"""

TYPE_ARCH = "1default" #"2def_drop" # "3def_bathNorm" #"1default"  #
DIM_LAST_LAYER = 4096 if TYPE_ARCH == "4AlexNet" else 512

P_DROPOUT = 0.2  # Probability of each element to be dropped
WITH_NORM_BATCH = False
DIST_THRESHOLD = 0.02
MARGIN = 2

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================
#                   CLASS: DistanceBased_Net
# ================================================================
class DistanceBased_Net(nn.Module):
    def __init__(self):
        super(DistanceBased_Net, self).__init__()

        if TYPE_ARCH == "1default":
            self.embedding_net = BasicNet()
        elif TYPE_ARCH == "4AlexNet":
            self.embedding_net = AlexNet()

        self.dist_threshold = 0.02

        self.to(DEVICE)
    '''---------------------------- get_distance --------------------------------- 
       This function gives the distance between the pairs (anchor, positive)
       and (anchor negative) based on their feature representation.
       If positive is None, the function just returns the distance between the 
       anchor and the negative
       ------------------------------------------------------------------------'''

    def get_distance(self, anchor, negative, positive=None, as_embedding=False):
        # --- Derivation of the feature representation ---
        if not as_embedding:
            embedded_anchor = self.embedding_net(anchor)
            embedded_neg = self.embedding_net(negative)
        else:
            embedded_anchor = anchor
            embedded_neg = negative

        # --- Computation of the distance between them ---
        distance = f.pairwise_distance(embedded_anchor, embedded_neg, 2)

        if positive is None:
            return distance, None
        else:
            embedded_pos = self.embedding_net(positive)
            disturb = f.pairwise_distance(embedded_anchor, embedded_pos, 2)
            return distance, disturb

    '''-------------------------- forward --------------------------------- 
       This function predicts if 2 pairs of data (anch, pos) and
       (anch, neg) refers to the same person 
       IN: data: list of (batch_size x 3) tensors 
       OUT: list of (batch_size x 2) tensors, such that a score has 
       been assigned to class 0 and class 1, based on the distances between
       the feature representation of the data 
    ---------------------------------------------------------------------'''

    def forward(self, data):
        distance, disturb = self.get_distance(data[0], data[2], data[1])

        output_positive = torch.ones([distance.size()[0], 2], dtype=torch.float64).to(DEVICE)
        output_positive[disturb <= self.dist_threshold, 1] = 0
        output_positive[disturb > self.dist_threshold, 1] = 2

        output_negative = torch.ones([distance.size()[0], 2], dtype=torch.float64).to(DEVICE)
        output_negative[self.dist_threshold <= distance, 0] = 0
        output_negative[distance < self.dist_threshold, 0] = 2

        return output_positive, output_negative

    def predict(self, data):
        embedded1 = self.embedding_net([data[0]])
        embedded2 = self.embedding_net([data[1]])
        return self.output_from_embedding(embedded1, embedded2)

    def output_from_embedding(self, embedding1, embedding2):
        distance, _ = self.get_distance(embedding1, embedding2, as_embedding=True)
        output = torch.ones([distance.size()[0], 2], dtype=torch.float64).to(DEVICE)
        output[distance <= self.dist_threshold, 1] = 0
        output[distance > self.dist_threshold, 1] = 2

        if not torch.cuda.is_available():
            return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()
        else:
            return torch.squeeze(torch.argmax(output, dim=1)).cuda().item()

    '''----------- update_dist_threshold ----------------------------- 
       This function updates the distance threshold by avg: 
            - the distances separting faces of different people,
            - the distances separting faces of same people,
            - the previous threshold value 
    -----------------------------------------------------------------'''

    def update_dist_threshold(self, dista, distb):
        avg_dista = float(sum(dista)) / dista.size()[0]
        avg_distb = float(sum(distb)) / dista.size()[0]
        self.dist_threshold = (self.dist_threshold + avg_dista + avg_distb) / 3


# ================================================================
#                    CLASS: Tripletnet
# ================================================================


class Tripletnet(DistanceBased_Net):
    def __init__(self):
        super(Tripletnet, self).__init__()

    def get_loss(self, data, target, class_weights):
        criterion = torch.nn.MarginRankingLoss(margin=MARGIN)
        distance, disturb = self.get_distance(data[0], data[2], data[1])
        self.update_dist_threshold(distance, disturb)

        # 1 means, dista should be greater than distb
        target = torch.FloatTensor(distance.size()).fill_(1).to(DEVICE)
        return criterion(distance, disturb, target)


# ================================================================
#                    CLASS: ContrastiveLoss
# ================================================================


class ContrastiveLoss(DistanceBased_Net):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 0 if samples are from the same class and label == 1 otherwise
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.eps = 1e-9

    def get_loss(self, data, target, class_weights):
        # Increase the distance "expectation" when we need more differentiation
        factor = max(2, class_weights[1] - class_weights[0])
        distance, disturb = self.get_distance(data[0], data[2], data[1])
        loss_pos = (0.5 * disturb.pow(2))
        loss_neg = (0.5 * f.relu(factor*DIST_THRESHOLD - distance).pow(2)) # 2*DIST_THRESHOLD = choice
        # loss.mean() or loss.sum()

        self.update_dist_threshold(distance, disturb)
        return (class_weights[0] * loss_pos + class_weights[1] * loss_neg).mean()


# ============================================================================================
#                    CLASS: CenterLoss
# This class comes from
# https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
# ============================================================================================
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, feat_dim, num_classes=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        if torch.cuda.is_available():
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if torch.cuda.is_available():
            classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss

    def compute_center_loss(self, features, targets):
        features = features.view(features.size(0), -1)
        target_centers = self.centers[targets]
        criterion = torch.nn.MSELoss()
        center_loss = criterion(features, target_centers)
        return center_loss

    def update_center(self, features, targets, alpha=0.5):

        # implementation equation (4) in the center-loss paper
        features = features.view(features.size(0), -1)
        targets, indices = torch.sort(targets)
        target_centers = self.centers[targets]
        features = features[indices]

        delta_centers = target_centers - features
        uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

        uni_targets = uni_targets.to(DEVICE)
        indices = indices.to(DEVICE)

        delta_centers = torch.zeros(
            uni_targets.size(0), delta_centers.size(1)
        ).to(DEVICE).index_add_(0, indices, delta_centers)

        targets_repeat_num = uni_targets.size()[0]
        uni_targets_repeat_num = targets.size()[0]
        targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
        uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
        same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

        delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
        result = torch.zeros_like(self.centers)
        result[uni_targets, :] = delta_centers
        self.centers = self.centers - result[uni_targets, :]


# ================================================================
#                   CLASS: SoftMax_Net
# ================================================================

class SoftMax_Net(nn.Module):
    def __init__(self, with_center_loss=False):
        super(SoftMax_Net, self).__init__()

        if TYPE_ARCH == "1default":
            self.embedding_net = BasicNet()
        elif TYPE_ARCH == "4AlexNet":
            self.embedding_net = AlexNet()
        else:
            print("ERR: Not matching embedding network!")
            raise Exception

        self.final_layer = nn.Linear(DIM_LAST_LAYER, 2).to(DEVICE)  # 2 = nb_classes
        self.loss_cur = 0

        self.center_loss = CenterLoss(DIM_LAST_LAYER, 2) if with_center_loss else None

    def forward(self, data):
        # Computation of the difference of the 2 feature representations
        feature_repr_anch = self.embedding_net(data[0])
        feature_repr_pos = self.embedding_net(data[1])
        feature_repr_neg = self.embedding_net(data[2])
        disturb = torch.abs(feature_repr_pos - feature_repr_anch)
        distance = torch.abs(feature_repr_neg - feature_repr_anch)

        # ---------- Center Loss Consideration ----------------
        if self.center_loss is not None:
            self.loss_cur = 0
            target_pos = torch.zeros([distance.size()[0]], dtype=torch.float64).type(torch.LongTensor).to(DEVICE)
            target_neg = torch.ones([distance.size()[0]], dtype=torch.float64).type(torch.LongTensor).to(DEVICE)

            loss_neg = self.center_loss(distance, target_neg)
            loss_pos = self.center_loss(disturb, target_pos)

            self.loss_cur = loss_neg + loss_pos


        return self.final_layer(disturb), self.final_layer(distance)
        # return self.avg_val_tensor(difference).requires_grad_(True) #

    def predict(self, data):
        feature_repr1 = self.embedding_net(data[0])
        feature_repr2 = self.embedding_net(data[1])
        return self.output_from_embedding(feature_repr1, feature_repr2)

    def output_from_embedding(self, embedding1, embedding2):
        dist = torch.abs(embedding2 - embedding1)
        last_values = self.final_layer(dist)
        if not torch.cuda.is_available():
            return torch.squeeze(torch.argmax(last_values, dim=1)).cpu().item()
        else:
            return torch.squeeze(torch.argmax(last_values, dim=1)).cuda().item()

    def avg_val_tensor(self, init_tensor):
        new_content = []
        # Go through each batch
        for i, pred in enumerate(init_tensor):
            sum_el = 0
            for j, elem in enumerate(pred):
                sum_el += elem
            new_content.append([1, sum_el / (DIST_THRESHOLD * len(pred))])

        return torch.tensor(new_content)  # , grad_fn=<AddBackward0>)  #requires_grad=True)

    def get_loss(self, data, target, class_weights):

        target = target.type(torch.LongTensor).to(DEVICE)
        target_positive = torch.squeeze(target[:, 0])  # = Only 0 here
        target_negative = torch.squeeze(target[:, 1])  # = Only 1 here

        output_positive, output_negative = self.forward(data)
        loss_positive = f.cross_entropy(output_positive, target_positive)
        loss_negative = f.cross_entropy(output_negative, target_negative)
        #print("Losses are: " + str(str(float(self.loss_cur))) + " and " + str(float(class_weights[0] * loss_positive + class_weights[1] * loss_negative)))
        return class_weights[0] * loss_positive + class_weights[1] * loss_negative + self.loss_cur

    """ ----------------------  visualize_last_output --------------------------------
    REM: would be more relevant if the pairs of point could be identified 
    -------------------------------------------------------------------------------- """
    def visualize_last_output(self, data, name_fig):
        out_pos, out_neg = self.forward(data)
        x = list(np.array(out_pos[:, 0].detach().cpu()))
        y = list(np.array(out_pos[:, 1].detach().cpu()))

        x.extend(list(np.array(out_neg[:, 0].detach().cpu())))
        y.extend(list(np.array(out_neg[:, 1].detach().cpu())))

        color0 = ["red" for i in range(len(out_pos[:, 0]))]
        color1 = ["green" for i in range(len(out_neg[:, 0]))]

        plt.show()
        plt.scatter(x, y, color=color0+color1)
        plt.show()
        plt.savefig(name_fig)

# ================================================================
#                    CLASS: BasicNet
# Initial Basic Network that was trained
# ================================================================

class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()

        # ----------- For Feature Representation -----------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7) #, padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(256, DIM_LAST_LAYER)
        self.dropout = nn.Dropout(P_DROPOUT)
        self.to(DEVICE)


        # Last layer assigning a number to each class from the previous layer
        # self.linear2 = nn.Linear(DIM_LAST_LAYER, 2)

    def forward(self, data):
        x = self.conv1(data.to(DEVICE))
        if WITH_NORM_BATCH: x = self.conv1_bn(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.pool4(x)

        x = self.conv2(x)
        if WITH_NORM_BATCH: x = self.conv2_bn(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.pool4(x)

        x = self.conv3(x)
        if WITH_NORM_BATCH: x = self.conv3_bn(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.pool4(x)

        x = x.view(x.shape[0], -1)  # To reshape
        x = self.linear1(x)
        return f.relu(x)


# ================================================================
#                    CLASS: AlexNet
# ================================================================

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=11, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(384, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
        )
        self.linearization = nn.Sequential(
            nn.Dropout(),
            nn.Linear(22 * 32 * 32, DIM_LAST_LAYER),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(DIM_LAST_LAYER, DIM_LAST_LAYER),
            nn.ReLU(inplace=True),
        )
        self.to(DEVICE)
        # self.final_layer = nn.Linear(DIM_LAST_LAYER, num_classes)

    def forward(self, data):
        x = self.features(data.to(DEVICE))
        x = x.view(x.size(0), 22 * 32 * 32) # 720896 / 32 = 22528.0
        return self.linearization(x)

# ================================================================
#                    CLASS: DecoderNet
# ================================================================

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.nb_channels = 4
        self.dim1 = 11
        self.dim2 = 11 + (CENTER_CROP[1] - CENTER_CROP[0])

        self.linear1 = nn.Linear(DIM_LAST_LAYER, self.nb_channels*self.dim1*self.dim2)
        self.conv3 = nn.ConvTranspose2d(self.nb_channels, 3, CENTER_CROP[0] - (self.dim1 - 1))
        self.sig = nn.Sigmoid()  # compress to a range (0, 1)
        self.to(DEVICE)

    def forward(self, data):
        x = self.linear1(data)
        x = x.view(x.size(0), self.nb_channels, self.dim1, self.dim2)

        x = self.conv3(x)
        x = self.sig(x)

        return x


# ================================================================
#                    CLASS: AutoEncoder
# ================================================================
class AutoEncoder(nn.Module):
    def __init__(self, embeddingNet):
        super(AutoEncoder, self).__init__()

        self.encoder = embeddingNet
        self.decoder = DecoderNet()
        self.last_decoded = None
        self.to(DEVICE)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        self.last_decoded = decoded
        return encoded, decoded

    def visualize_dec(self):
        for i, decoded in enumerate(self.last_decoded[0].detach()):
            dec_as_np = decoded.cpu().numpy()
            plt.imshow(np.transpose(dec_as_np))
            print("The picture representing the result from the decoder is saved")
            plt.savefig("Result_autoencoder_" + str(i))
            plt.show()
