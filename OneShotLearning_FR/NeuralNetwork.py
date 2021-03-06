import torch
from torch import nn
import torch.nn.functional as f
import numpy as np

from Dataprocessing import CENTER_CROP, Face_DS, from_zip_to_data
from EmbeddingNetwork import AlexNet, BasicNet, VGG16

from scipy.misc import toimage

import matplotlib

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
5: VGG16 Net 
"""

TYPE_ARCH = "1default"  # "4AlexNet" "resnet152" "VGG16" #  "2def_drop" "3def_bathNorm"
DIM_LAST_LAYER = 1024 if TYPE_ARCH in ["VGG16", "4AlexNet", "1default1024"] else 512

DIST_THRESHOLD = 0.02
MARGIN = 0.2
METRIC = "Euclid"  # "Cosine" #
NORMALIZE_DIST = False
NORMALIZE_FR = True
WEIGHT_DIST = 0.2  # for dist_loss
WITH_DIST_WEIGHT = False

if TYPE_ARCH in ["4AlexNet", "VGG16"]:
    NORMALIZE_FR = False

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
#                   Functions
# ================================================================

"""
IN: tensor: a 1D-tensor 
OUT: the median computed from the elements of this tensor 
"""


def get_median(tensor):
    sorted_elements, _ = torch.sort(tensor)
    return sorted_elements[round(tensor.size()[0] / 2)].item()


# ================================================================
#                   CLASS: DistanceBased_Net
# ================================================================
class DistanceBased_Net(nn.Module):
    def __init__(self, embeddingNet, metric):
        super(DistanceBased_Net, self).__init__()

        if embeddingNet is not None:
            self.embedding_net = embeddingNet
        elif TYPE_ARCH == "1default" or TYPE_ARCH == "1default1024":
            self.embedding_net = BasicNet(DIM_LAST_LAYER)
        elif TYPE_ARCH == "4AlexNet" or TYPE_ARCH == "4AlexNet2048":
            self.embedding_net = AlexNet(DIM_LAST_LAYER)
        elif TYPE_ARCH == "VGG16":
            self.embedding_net = VGG16(DIM_LAST_LAYER)
        elif TYPE_ARCH[:len("resnet")] == "resnet":
            self.embedding_net = ResNet(DIM_LAST_LAYER, resnet=TYPE_ARCH)
        else:
            print("ERR: No matching with the given architecture...")
            raise Exception

        self.dist_threshold = DIST_THRESHOLD

        # To weights the distance elements
        self.final_layer = nn.Linear(DIM_LAST_LAYER, DIM_LAST_LAYER)  # .to(DEVICE)

        self.metric = metric
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
            if NORMALIZE_FR:
                embedded_anchor = f.normalize(embedded_anchor, p=2, dim=1)
                embedded_neg = f.normalize(embedded_neg, p=2, dim=1)
        else:
            embedded_anchor = anchor
            embedded_neg = negative

        # --- Computation of the distance between them ---
        distance = None
        if not WITH_DIST_WEIGHT or positive is None:
            distance = f.pairwise_distance(embedded_anchor, embedded_neg, 2) if self.metric == "Euclid" \
                else f.cosine_similarity(embedded_anchor, embedded_neg)

        # ------------------------------------------------------
        # CASE 1: Just get the distance between 2 input
        # ------------------------------------------------------
        if positive is None:
            if NORMALIZE_DIST:
                distance = abs(distance - distance.mean) / distance.std
            return distance, None

        # ---------------------------------------------------------------------
        # CASE 2: Get the distance and the disturbance related to (A,P,N)
        # ---------------------------------------------------------------------
        else:
            embedded_pos = self.embedding_net(positive)
            if NORMALIZE_FR:
                embedded_pos = f.normalize(self.embedding_net(positive), p=2, dim=1)

            if WITH_DIST_WEIGHT:
                disturb = torch.abs(embedded_anchor - embedded_pos)
                distance = torch.abs(embedded_anchor - embedded_neg)
                distance = self.final_layer(distance).mean(-1)
                disturb = self.final_layer(disturb).mean(-1)
                return distance, disturb

            disturb = f.pairwise_distance(embedded_anchor, embedded_pos, 2) if self.metric == "Euclid" \
                else f.cosine_similarity(embedded_anchor, embedded_pos)

            if NORMALIZE_DIST:
                distance = abs(distance - torch.mean(distance)) / torch.std(distance)
                disturb = abs(disturb - torch.mean(disturb)) / torch.std(disturb)

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
        output_positive[disturb < self.dist_threshold, 1] = 0
        output_positive[disturb >= self.dist_threshold, 1] = 2

        output_negative = torch.ones([distance.size()[0], 2], dtype=torch.float64).to(DEVICE)
        output_negative[self.dist_threshold <= distance, 0] = 0
        output_negative[distance < self.dist_threshold, 0] = 2

        return output_positive, output_negative

    '''-------------------------- predict --------------------------------- '''

    def predict(self, data):
        embedded1 = self.embedding_net(data[0])
        embedded2 = self.embedding_net(data[1])
        if NORMALIZE_FR:
            embedded1 = f.normalize(embedded1, p=2, dim=1)
            embedded2 = f.normalize(embedded2, p=2, dim=1)

        return self.output_from_embedding(embedded1, embedded2)

    '''----------------- output_from_embedding -------------------------------- '''

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

        med_dista = get_median(dista)  # avg_dista = float(sum(dista)) / dista.size()[0]
        med_distb = get_median(distb)  # avg_distb = float(sum(distb)) / dista.size()[0]

        self.dist_threshold = (med_dista + med_distb) / 2
        return med_dista, med_distb


# ================================================================
#                    CLASS: Triplet_Net
# ================================================================


class Triplet_Net(DistanceBased_Net):
    def __init__(self, embedding_net, with_dist_loss=False, with_ce_loss=False, metric=METRIC):
        super(Triplet_Net, self).__init__(embedding_net, metric)
        self.margin = MARGIN
        self.with_dist_loss = with_dist_loss
        self.with_ce_loss = with_ce_loss

    '''------------------ get_loss ------------------------ '''

    def get_loss(self, data, target, class_weights, train=True):

        # ----------------------------------
        # Triplet Loss Definition
        # Aim = Minimize (disturb - distance) (and try it to be less than the margin)
        # Risk = Dist close to distance => interest of Dist Loss
        # ----------------------------------

        criterion = torch.nn.MarginRankingLoss(margin=self.margin)  # CosineEmbeddingLoss

        distance, disturb = self.get_distance(data[0], data[2], data[1])
        if train:
            med_distance, med_disturb = self.update_dist_threshold(distance, disturb)
        else:
            med_distance = get_median(distance)
            med_disturb = get_median(disturb)

            if abs(med_distance - med_disturb) < self.margin:
                pass  # Way to adapt loss according to the current state

        # 1 means, dista should be greater than distb
        target_triplet = torch.FloatTensor(distance.size()).fill_(1).to(DEVICE)
        loss = criterion((1 / class_weights[1]) * distance, class_weights[0] * disturb, target_triplet)

        # ----------------------------------
        # "Dist Loss" Definition
        # Explicit prevention from having same distance and disturbance leading to a constant loss (= margin)
        # AIM = Maximize gap between the distance and the disturbance
        # ----------------------------------
        if self.with_dist_loss:
            #
            diff_dist_loss_obj = torch.nn.L1Loss()
            diff_dist_loss = diff_dist_loss_obj(distance, disturb)
            # ("diff_dist_loss " + str(diff_dist_loss.item()))

            # Distance = Disturbance Case Detection
            if diff_dist_loss.item() < 0.01:
                loss = (min(1 / med_distance, 100) + med_disturb) / 2 - diff_dist_loss
            else:
                loss = (1 - WEIGHT_DIST) * loss - WEIGHT_DIST * diff_dist_loss
        # ----------------------------------
        # Cross Entropy Loss Definition
        # ----------------------------------
        elif self.with_ce_loss:
            target = target.type(torch.LongTensor).to(DEVICE)
            target_positive = torch.squeeze(target[:, 0])  # = Only 0 here
            target_negative = torch.squeeze(target[:, 1])  # = Only 1 here

            output_positive, output_negative = self.forward(data)
            loss_positive = f.cross_entropy(output_positive, target_positive).float().cuda()
            loss_negative = f.cross_entropy(output_negative, target_negative).float().cuda()
            loss += loss_positive + loss_negative

        return loss


# ================================================================
#                    CLASS: ContrastiveLoss
# ================================================================


class ContrastiveLoss(DistanceBased_Net):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 0 if samples are from the same class and label == 1 otherwise
    """

    def __init__(self, embeddingNet):
        super(ContrastiveLoss, self).__init__(embeddingNet, METRIC)
        self.eps = 1e-9

    '''------------------ get_loss ------------------------ '''

    def get_loss(self, data, target, class_weights, train=True):
        # Increase the distance "expectation" when we need more differentiation
        factor = max(2, class_weights[1] - class_weights[0])
        distance, disturb = self.get_distance(data[0], data[2], data[1])
        loss_pos = (0.5 * disturb.pow(2))
        loss_neg = (0.5 * f.relu(factor * DIST_THRESHOLD - distance).pow(2))  # 2*DIST_THRESHOLD = choice
        # loss.mean() or loss.sum()

        if train: self.update_dist_threshold(distance, disturb)
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

        self.to(DEVICE)

    '''------------------ forward ------------------------ '''

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
        arg = classes.expand(batch_size, self.num_classes)
        mask = labels.eq(arg)

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
#                   CLASS: Classif_Net
# ================================================================

class Classif_Net(nn.Module):
    def __init__(self, embeddingNet, nb_classes=10, with_center_loss=True):
        super(Classif_Net, self).__init__()

        if embeddingNet is not None:
            self.embedding_net = embeddingNet
        elif TYPE_ARCH == "1default":
            self.embedding_net = BasicNet(DIM_LAST_LAYER)
        elif TYPE_ARCH == "4AlexNet" or TYPE_ARCH == "4AlexNet2048":
            self.embedding_net = AlexNet(DIM_LAST_LAYER)
        elif TYPE_ARCH == "VGG16":
            self.embedding_net = VGG16(DIM_LAST_LAYER)
        elif TYPE_ARCH[:len("resnet")] == "resnet":
            self.embedding_net = ResNet(DIM_LAST_LAYER, resnet=TYPE_ARCH)
        else:
            print("ERR: Not matching embedding network!")
            raise Exception

        self.final_layer = nn.Linear(DIM_LAST_LAYER, nb_classes)  # .to(DEVICE)
        self.loss_cur = 0
        self.center_loss = CenterLoss(DIM_LAST_LAYER, nb_classes) if with_center_loss else None
        self.to(DEVICE)

    def forward(self, data, label):
        embedded_data = self.embedding_net(data)
        if NORMALIZE_FR:
            embedded_data = f.normalize(embedded_data, p=2, dim=1)

        # ---------- Center Loss Consideration ----------------
        if self.center_loss is not None:
            self.loss_cur = self.center_loss(embedded_data, label)

        return self.final_layer(embedded_data)

    def predict(self, data):
        feature_repr = self.embedding_net(data)
        if NORMALIZE_FR:
            feature_repr = f.normalize(feature_repr, p=2, dim=1)

        return self.output_from_embedding(feature_repr)

    def output_from_embedding(self, embedding):
        last_values = self.final_layer(embedding)
        if not torch.cuda.is_available():
            return torch.squeeze(torch.argmax(last_values, dim=1)).cpu().item()
        else:
            return torch.squeeze(torch.argmax(last_values, dim=1)).cuda().item()

    def get_loss(self, data, target, class_weights, train=True):
        outputs = self.forward(data, target)
        loss = f.cross_entropy(outputs, target) + self.loss_cur
        return loss


# ================================================================
#                   CLASS: SoftMax_Net
# ================================================================

class SoftMax_Net(nn.Module):
    def __init__(self, embeddingNet, with_center_loss=False, nb_classes=2):
        super(SoftMax_Net, self).__init__()

        if embeddingNet is not None:
            self.embedding_net = embeddingNet
            # DIM_LAST_LAYER = 1024 if embeddingNet.name_arch in ["VGG16", "4AlexNet", "resnet"] else 512
        elif TYPE_ARCH == "1default" or TYPE_ARCH == "1default1024":
            self.embedding_net = BasicNet(DIM_LAST_LAYER)
        elif TYPE_ARCH == "4AlexNet" or TYPE_ARCH == "4AlexNet2048":
            self.embedding_net = AlexNet(DIM_LAST_LAYER)
        elif TYPE_ARCH == "VGG16":
            self.embedding_net = VGG16(DIM_LAST_LAYER)
        elif TYPE_ARCH[:len("resnet")] == "resnet":
            self.embedding_net = ResNet(DIM_LAST_LAYER, resnet=TYPE_ARCH)
        else:
            print("ERR: No matching with the given architecture...")
            raise Exception

        self.final_layer = nn.Linear(DIM_LAST_LAYER, nb_classes)  # .to(DEVICE)

        self.loss_cur = 0
        self.center_loss = CenterLoss(DIM_LAST_LAYER, nb_classes) if with_center_loss else None
        self.to(DEVICE)

    def forward(self, data):
        # Computation of the difference of the 2 feature representations
        feature_repr_anch = self.embedding_net(data[0])
        feature_repr_pos = self.embedding_net(data[1])
        feature_repr_neg = self.embedding_net(data[2])
        if NORMALIZE_FR:
            feature_repr_anch = f.normalize(feature_repr_anch, p=2, dim=1)
            feature_repr_pos = f.normalize(feature_repr_pos, p=2, dim=1)
            feature_repr_neg = f.normalize(feature_repr_neg, p=2, dim=1)

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
        embedded1 = self.embedding_net(data[0])
        embedded2 = self.embedding_net(data[1])
        if NORMALIZE_FR:
            embedded1 = f.normalize(embedded1, p=2, dim=1)
            embedded2 = f.normalize(embedded2, p=2, dim=1)
        return self.output_from_embedding(embedded1, embedded2)

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

    def get_loss(self, data, target, class_weights, train=True):

        target = target.type(torch.LongTensor).to(DEVICE)
        target_positive = torch.squeeze(target[:, 0])  # = Only 0 here
        target_negative = torch.squeeze(target[:, 1])  # = Only 1 here

        output_positive, output_negative = self.forward(data)
        loss_positive = f.cross_entropy(output_positive, target_positive)
        loss_negative = f.cross_entropy(output_negative, target_negative)
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
        plt.scatter(x, y, color=color0 + color1)
        plt.show()
        plt.savefig(name_fig)


# ================================================================
#                    CLASS: DecoderNet
# ================================================================

class DecoderNet(nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.nb_channels = 4
        self.out_nb_channels = 3
        self.dim1 = 140
        self.dim2 = self.dim1 + (CENTER_CROP[1] - CENTER_CROP[0])  # = 150

        self.linear1 = nn.Linear(DIM_LAST_LAYER, self.nb_channels * self.dim1 * self.dim2)
        self.conv3 = nn.ConvTranspose2d(self.nb_channels, self.out_nb_channels, CENTER_CROP[0] - (self.dim1 - 1))
        self.conv4 = nn.ConvTranspose2d(self.out_nb_channels, self.out_nb_channels, 6, stride=2, padding=2)
        self.conv5 = nn.ConvTranspose2d(self.out_nb_channels, self.out_nb_channels, 6, stride=2, padding=2)

        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU(True)

        self.sig = nn.Sigmoid()  # compress to a range (0, 1)
        self.to(DEVICE)

    def forward(self, data):
        x = self.linear1(data)
        x = x.view(x.size(0), self.nb_channels, self.dim1, self.dim2)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)

        x = self.sig(x)
        x = x.view(x.size(0), self.out_nb_channels, CENTER_CROP[0],
                   CENTER_CROP[1])  # 3 * 200 * 150 = 90 000  * 32 = 2 880 000

        return x


# ================================================================
#                    CLASS: AutoEncoder_Net
# ================================================================
class AutoEncoder_Net(nn.Module):
    def __init__(self, embeddingNet):
        super(AutoEncoder_Net, self).__init__()

        self.encoder = embeddingNet
        self.decoder = DecoderNet()
        self.last_decoded = None
        self.to(DEVICE)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        self.last_decoded = decoded

        return encoded, decoded

    def visualize_dec(self, epoch=180):
        dec_as_np = self.last_decoded[0].detach().cpu().numpy()

        plt.imshow(np.reshape(dec_as_np, [200, 150, 3]))
        print("The picture representing the result from the decoder is saved as " + "resAut_" + TYPE_ARCH)
        plt.savefig("resAuto_" + TYPE_ARCH + "_" + str(epoch))  # .squeeze()

        toimage(dec_as_np).save("resAutScip_" + TYPE_ARCH + "_" + str(epoch) + ".png", "png")


# ================================================================
#                    FUNCTIONS
# ================================================================

def visualize_autoencoder_res(name_autoencoder, db_source):
    # ----------- Load autoencoder --------------------
    if name_autoencoder.split(".")[-1] == "pwf":
        network = Triplet_Net(None)
        autoencoder = AutoEncoder_Net(network.embedding_net)
        autoencoder.load_state_dict(torch.load(name_autoencoder))
    else:
        print("IN VISUALIZE AUTOENCODER: Wrong format ....")
        return

    # ----------- Set last_decoded --------------------
    fileset = from_zip_to_data(False, db_source)
    ds = Face_DS(fileset=fileset, triplet_version=False)
    train_loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        autoencoder(data.to(DEVICE))
    autoencoder.visualize_dec()


# ================================================================
#                    MAIN
# ================================================================

if __name__ == '__main__':
    name_autoencoder = "encoders/auto_180.pwf"
    db_source = "/data/gbrieven/cfp_3.zip"
