# Source: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# Source: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

import math
import torch
from torch import nn
import torch.nn.functional as f

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

P_DROPOUT = 0.2  # Probability of each element to be dropped (default value is 0.5)
WITH_NORM_BATCH = False
BATCH_SIZE = 32

# Specifies where the torch.tensor is allocated
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WITH_GNAP = False
W = 7


# ================================================================
#                    CLASS: BasicNet
# Initial Basic Network that was trained
# ================================================================

class BasicNet(nn.Module):
    def __init__(self, dim_last_layer):
        super(BasicNet, self).__init__()
        self.name_arch = "1default"

        # ----------- For Feature Representation -----------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(P_DROPOUT)
        self.to(DEVICE)

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
    def __init__(self, dim_last_layer):
        super(AlexNet, self).__init__()
        self.name_arch = "4AlexNet"
        if WITH_GNAP: print("The GNAP module is used\n")
        self.dim_last_layer = dim_last_layer

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=7, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=2),
            nn.Conv2d(384, 256, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.gnap = GNAP()
        self.linearization = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, dim_last_layer),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(dim_last_layer, dim_last_layer),  # Free first dim
            nn.ReLU(inplace=True)
        )
        self.to(DEVICE)

    def forward(self, data):
        x = self.features(data.to(DEVICE))

        if WITH_GNAP: x = self.gnap(x)
        x = x.view(x.size(0), 512)  # 16384 /32 = 512.0

        return self.linearization(x)


# ================================================================
#                    CLASS: VGG16
# ================================================================



class VGG16(nn.Module):
    def __init__(self, dim_last_layer, init_weights=True):  # num_class potentially to modify

        super(VGG16, self).__init__()
        self.name_arch = "VGG16"
        if WITH_GNAP: print("The GNAP module is used\n")

        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = make_layers(cfg, batch_norm=WITH_NORM_BATCH)

        self.gnap = GNAP()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.linearization = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, dim_last_layer),
        )

        if init_weights:
            self._initialize_weights()

        self.to(DEVICE)

    def forward(self, data):
        x = self.features(data.to(DEVICE))

        if WITH_GNAP: x = self.gnap(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return self.linearization(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# ================================================================
#                    CLASS: BasicBlock
# ================================================================


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


# ================================================================
#                    CLASS:  GNAP block
#               (Global Norm-Aware Pooling)
# ================================================================

class GNAP(nn.Module):
    def __init__(self, in_dim=256):
        super(GNAP, self).__init__()

        # ----------- Batch Normalization -----------------
        self.batchNorm = nn.BatchNorm2d(in_dim)
        self.H = 2048
        self.W = W
        self.C = 5

        # ----------- Global Average Pooling -----------------
        self.globAvgPool = nn.AdaptiveAvgPool2d(self.W)

    '''-------------------- normAwareReweighting ------------------------ 
       This function acts a a norm-aware reweighting layer
       ASSUMPTION: input of dimensions C x W x H 
       ----------------------------------------------------------------'''

    def normAwareReweighting(self, x):
        batch_size = x.size()[0]
        fij = []

        # ------ STEP 1: L2 Norm of the local feature -------
        for b in range(batch_size):
            fij.append([])
            for h in range(self.H):
                for w in range(self.W):
                    fij[b].append(0)
                    for channel in range(self.C):
                        fij[b][-1] += x[b][h][w][channel] ** 2
                    fij[b][-1] = math.sqrt(fij[b][-1])

            # ------ STEP 2: Mean of local features' L2 norms -------
            f_mean = sum(fij[b]) / (self.H * self.W)

            # ------ STEP 3: Norm-aware Reweighting -------
            i = 0
            for h in range(self.H):
                for w in range(self.W):
                    for channel in range(self.C):
                        x[b][h][w][channel] = (f_mean / fij[b][i]) * x[b][h][w][channel]
                    i += 1
        return x

    def forward(self, init_feature_repres):
        if not WITH_GNAP:
            return init_feature_repres
        else:
            x = self.batchNorm(init_feature_repres)
            x = self.normAwareReweighting(x)
            x = self.globAvgPool(x)
            return self.batchNorm(x)
