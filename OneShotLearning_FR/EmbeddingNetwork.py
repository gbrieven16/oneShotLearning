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
#LAST_DIM = 512

# For ResNet:
LAYERS_RES = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3], "resnet50": [3, 4, 6, 3],
              "resnet101": [3, 4, 23, 3], "resnet152": [3, 8, 36, 3]}

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7)  # , padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(4)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.linear1 = nn.Linear(256, dim_last_layer)
        self.dropout = nn.Dropout(P_DROPOUT)
        self.to(DEVICE)


        # Last layer assigning a number to each class from the previous layer
        # self.linear2 = nn.Linear(dim_last_layer, 2)

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
            nn.Linear(dim_last_layer, dim_last_layer), # Free first dim
            nn.ReLU(inplace=True)
        )
        self.to(DEVICE)
        # self.final_layer = nn.Linear(dim_last_layer, num_classes)

    def forward(self, data):
        #x = self.features(data)
        x = self.features(data.to(DEVICE))
        if WITH_GNAP: x = self.gnap(x)
        x = x.view(x.size(0), 512) #16384 /32 = 512.0

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

    def forward(self, data):
        x = self.features(data.to(DEVICE))
        if WITH_GNAP: x = self.gnap(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linearization(x)
        return x

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
#                    CLASS: VGG16
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
        out = self.relu(out)
        return out

# ================================================================
#                    CLASS: ResNet
# ================================================================


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, dim_last_layer, zero_init_residual=False, resnet="resnet152"):
        super(ResNet, self).__init__()
        if WITH_GNAP: print("The GNAP module is used\n")
        self.name_arch = "resnet"
        if resnet == "resnet18" or resnet == "resnet34":
            block = BasicBlock
        else:
            block = Bottleneck

        layers = LAYERS_RES[resnet]

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.gnap = GNAP(in_dim=512 * block.expansion)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, dim_last_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       nn.BatchNorm2d(planes * block.expansion),
                                       )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gnap(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


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
                        fij[b][-1] += x[b][h][w][channel]**2
                    fij[b][-1] = math.sqrt(fij[b][-1])

            # ------ STEP 2: Mean of local features' L2 norms -------
            f_mean = sum(fij[b])/(self.H*self.W)

            # ------ STEP 3: Norm-aware Reweighting -------
            i = 0
            for h in range(self.H):
                for w in range(self.W):
                    for channel in range(self.C):
                        x[b][h][w][channel] = (f_mean/fij[b][i]) * x[b][h][w][channel]
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


