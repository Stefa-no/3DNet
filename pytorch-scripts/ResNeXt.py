from __future__ import division
""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
import from https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
DOWNLOADED FROM https://github.com/bearpaw/pytorch-classification/blob/master/models/imagenet/resnext.py
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['resnext50', 'resnext101', 'resnext152']

class Bottleneck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, drop_rate=0, preactivation=True, stochastic=True, personalized=True, activ_fun='relu'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D*C)
        self.bn1p = nn.BatchNorm2d(inplanes)
        self.conv2a = nn.Conv2d(D*C, D*C//2, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.conv2b = nn.Sequential(
                nn.Conv2d(D*C, D*C, kernel_size=3, stride=1, padding=1, groups=C, bias=False),
                nn.Conv2d(D*C, D*C//2, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
                )
        self.conv2c = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D*C)
        self.bn2p = nn.BatchNorm2d(D*C)
        self.conv3 = nn.Conv2d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3p = nn.BatchNorm2d(D*C)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.activ_fun = activ_fun
        
        
        if activ_fun == 'relu':
                self.relu = nn.ReLU(inplace=True)
        else:
                self.relu = nn.ReLU6(inplace=True)
        self.stocdepth = nn.Dropout(p=drop_rate)
        self.activ_fun = activ_fun

        self.downsample = downsample
        self.preactivation = preactivation
        self.stochastic = stochastic
        self.personalized = personalized
        

    def forward(self, x):
        residual = x
        if self.preactivation:
                out = self.bn1p(x)
                out = self.relu(out)
                out = self.conv1(out)

                if self.personalized:
                        out = self.bn2p(out)
                        out = self.relu(out)
                        outa = self.conv2a(out)
                        outb = self.conv2b(out)
                        out = torch.cat( (outa, outb), 1)
                else:
                        out = self.bn2p(out)
                        out = self.relu(out)
                        out = self.conv2c(out)

                out = self.bn3p(out)
                out = self.relu(out)
                out = self.conv3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)   
                
                if self.stochastic: out = self.stocdepth(out)     

                out += residual
                                
        else:
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                if self.personalized:
                        outa = self.conv2a(out)
                        outb = self.conv2b(out)
                        out = torch.cat( (outa, outb), 1)
                        out = self.bn2(out)
                        out = self.relu(out)
                else:
                        out = self.conv2c(out)
                        out = self.bn2(out)
                        out = self.relu(out)
                        
                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)   
                
                if self.stochastic: out = self.stocdepth(out)     

                out += residual
                out = self.relu(out)
                
        return out


class ResNeXt(nn.Module):
    """
    ResNext optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, baseWidth, cardinality, layers, num_classes, preactivation=True, stochastic=True, personalized=True, activ_fun='relu'):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()
        block = Bottleneck

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64
        self.depth = sum(layers)
        self.preactivation = preactivation
        self.stochastic = stochastic
        self.personalized = personalized
        self.activ_fun = activ_fun

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if activ_fun == 'relu':
                self.relu = nn.ReLU(inplace=True)
        else:
                self.relu = nn.ReLU6(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        
        rel_depth = 0
        self.layer1 = self._make_layer(block, 64, layers[0], rel_depth,  1, self.preactivation, self.stochastic, self.personalized)
        rel_depth += layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], rel_depth, 2, self.preactivation, self.stochastic, self.personalized)
        rel_depth += layers[1]
        self.layer3 = self._make_layer(block, 256, layers[2], rel_depth, 2, self.preactivation, self.stochastic, self.personalized)
        rel_depth += layers[2]
        self.layer4 = self._make_layer(block, 512, layers[3], rel_depth, 2, self.preactivation, self.stochastic, self.personalized)
        self.avgpool = nn.AvgPool2d(7)      
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, layernum, stride=1, preactivation=True, stochastic=True, personalized=True):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
            layernum: the depth of the current layer
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        drop_rate = 1 - (layernum / self.depth) * ( 1 - 0.5)

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample, drop_rate=drop_rate, preactivation=preactivation, stochastic=stochastic, personalized=personalized, activ_fun=self.activ_fun))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            drop_rate = 1 - ( (layernum+i) / self.depth) * ( 1 - 0.5)
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, drop_rate=drop_rate, preactivation=preactivation, stochastic=stochastic, personalized=personalized, activ_fun=self.activ_fun))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext18(baseWidth, cardinality, num_classes=200, preactivation=True, stochastic=True, personalized=True, activ_fun='relu'):
    """
    Construct ResNeXt-50.
    """
    model = ResNeXt(baseWidth, cardinality, [1,1,1,1], num_classes, preactivation, stochastic, personalized, activ_fun)
    return model

