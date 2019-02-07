"""
Parts of this file are taken from Soumith Chintalas implementation [1] and 
derived from it, respecitively. His implementation was licensed under the 
BSD 3-Clause License (see [2]). Hence, for this two licenses apply 
at the same time: The BSD 3-Clause and GPLv3.

For the original code:
Copyright (c) Soumith Chintala 2016

For the  new code:
Copyright 2018 ICG, Graz University of Technology

This file is part of MURAUER.

MURAUER is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MURAUER is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MURAUER.  If not, see <http://www.gnu.org/licenses/>.

[1] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
[2] https://github.com/pytorch/vision/blob/master/LICENSE
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def conv3x3(num_in_planes, num_out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(num_in_planes, num_out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, num_in_planes, num_out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(num_in_planes, num_out_planes, stride)
        self.bn1 = nn.BatchNorm2d(num_out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_out_planes, num_out_planes)
        self.bn2 = nn.BatchNorm2d(num_out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_in_planes, num_bottleneck_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(num_in_planes, num_bottleneck_planes, 
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_bottleneck_planes)
        self.conv2 = nn.Conv2d(num_bottleneck_planes, num_bottleneck_planes, 
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_bottleneck_planes)
        self.conv3 = nn.Conv2d(num_bottleneck_planes, num_bottleneck_planes * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_bottleneck_planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MappingBlock(nn.Module):
    """
    As described in Rad et al., CVPR 2018 [1]
    
    [1] https://arxiv.org/abs/1712.03904
    """
    expansion = 1

    def __init__(self, num_in_planes, num_out_planes, stride=1, downsample=None):
        super(MappingBlock, self).__init__()
        self.fc1 = nn.Linear(num_in_planes, num_out_planes)
        self.fc2 = nn.Linear(num_in_planes, num_out_planes)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
        

#%% Network definition
class DiscriminatorResNet(nn.Module):
    """
    Discriminator (for an embedding) based on ResNet blocks
    """
    
    def __init__(self, num_in_planes, num_features=1024, 
                 base_block=MappingBlock, num_blocks=2):
        """
        
        Arguments:
            num_features (int, optional): number of input features/dimensions
                default: 1024
        """
        super(DiscriminatorResNet, self).__init__()
        
        nf = num_features
        
        # ResNet
        self.inplanes = num_in_planes
        self.resnetlayer = self._make_layer(base_block, nf, num_blocks)
        
        self.fc = nn.Linear(nf, 1)
        
        
    def _make_layer(self, base_block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * base_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * base_block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * base_block.expansion),
            )

        layers = []
        layers.append(base_block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * base_block.expansion
        for i in range(1, blocks):
            layers.append(base_block(self.inplanes, planes))

        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.resnetlayer(x)
        x = self.fc(x)
        return x
        
        
class PreViewDecoderDcGan(nn.Module):
    """
    Decoder net, based on DC-GAN architecture [1].
    
    [1] Radford et al., ICLR 2016, https://arxiv.org/pdf/1511.06434.pdf
    """
    
    def __init__(self, num_input_dim=30, num_com_dim=3, num_features=64):
        """
        
        Arguments:
            do_use_gpu (boolean, optional): compute on GPU (=default) or CPU?
            num_input_dim (int, optional): #input dimensions; default: 30
            num_com_dim (int, optional): #dimensions for com input; default: 3
            num_features (int, optional): number of feature channels in the 
                first/lowest layer (highest resolution), it is increased 
                inversely proportional with downscaling at each layer; 
                default: 64
        """
        super(PreViewDecoderDcGan, self).__init__()
        
        nf = num_features

        # Decoder
        self.convt1 = nn.ConvTranspose2d(num_input_dim + num_com_dim, 
                                         (nf * 8), (4, 4), 
                                         stride=1, padding=0, bias=False)
        self.bn1_d = nn.BatchNorm2d(nf * 8)
        self.convt2 = nn.ConvTranspose2d((nf * 8), (nf * 4), (4, 4), 
                                         stride=2, padding=1, bias=False)
        self.bn2_d = nn.BatchNorm2d(nf * 4)
        self.convt3 = nn.ConvTranspose2d((nf * 4), (nf * 2), (4, 4), 
                                         stride=2, padding=1, bias=False)
        self.bn3_d = nn.BatchNorm2d(nf * 2)
        self.convt4 = nn.ConvTranspose2d((nf * 2), nf, (4, 4), 
                                         stride=2, padding=1, bias=False)
        self.bn4_d = nn.BatchNorm2d(nf)
        self.convt5 = nn.ConvTranspose2d(nf, 1, (4, 4), 
                                         stride=2, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        

    def decode(self, z):
        y = self.leakyrelu(self.bn1_d(self.convt1(z)))
        y = self.leakyrelu(self.bn2_d(self.convt2(y)))
        y = self.leakyrelu(self.bn3_d(self.convt3(y)))
        y = self.leakyrelu(self.bn4_d(self.convt4(y)))
        y = torch.tanh(F.interpolate(self.convt5(y), scale_factor=2, 
                                     mode='bilinear', align_corners=True))
        return y
        

    def forward(self, z, com):
        """
        Arguments:
            com (Tensor): conditional for generator/decoder; 
        """
        z = torch.cat((z, com), 1)      # concat. z and com
        z = z.unsqueeze(2).unsqueeze(2) # add two singleton dims; BxD => BxDx1x1
        y = self.decode(z)
        return y


class ResNet_Map_PreView(nn.Module):
    """
    ResNet based architecture for hand pose regression with 
    mapping layers as in Rad et al., CVPR 2018 [1], and 
    view prediction as in our CVPR 2018 paper [2].
    
    [1] https://arxiv.org/abs/1712.03904
    [2] https://arxiv.org/abs/1804.03390
    """

    def __init__(self, base_block, num_blocks_per_layer, num_classes=(16*3), 
                 num_features=32, mapping_block=MappingBlock, 
                 num_blocks_mappinglayer=2, num_bottleneck_dim=1024):
        """
        Arguments:
            num_features (int, optional): number of feature channels in the 
                first/lowest layer (highest resolution), it is increased 
                inversely proportional with downscaling at each layer; 
                default: 32
        """
        nf = num_features
        nb = num_bottleneck_dim
        self.inplanes = nf
        super(ResNet_Map_PreView, self).__init__()
        self.conv1 = nn.Conv2d(1, nf, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(base_block, nf*2 // 4, num_blocks_per_layer[0], stride=2)
        self.layer2 = self._make_layer(base_block, nf*4 // 4, num_blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(base_block, nf*8 // 4, num_blocks_per_layer[2], stride=2)
        self.layer4 = self._make_layer(base_block, nf*8 // 4, num_blocks_per_layer[3], stride=2)
        self.bn_res = nn.BatchNorm2d(nf*8)
        self.fc1 = nn.Linear(nf*8*4*4, 1024)
        self.fc2 = nn.Linear(1024, nb)
        
        # Mapping layer
        self.inplanes = 1024
        self.mappinglayer = self._make_layer(mapping_block, 1024, num_blocks_mappinglayer)
        
        self.preview = PreViewDecoderDcGan(nb, 3)
        
        self.fc3_h0 = nn.Linear(nb, num_classes)
        self.fc3_h1 = nn.Linear(nb, num_classes)    # for using pre-trained models

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0.0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, base_block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * base_block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * base_block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * base_block.expansion),
            )

        layers = []
        layers.append(base_block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * base_block.expansion
        for i in range(1, blocks):
            layers.append(base_block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, com, do_map=True, do_preview=True, 
                do_backprop_through_feature_extr_before_map=True):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.bn_res(x)
        x = self.relu(x)

        x = x.view(x.size(0), -1)
        emb = self.fc1(x)
        if do_map:
            if not do_backprop_through_feature_extr_before_map:
                emb.detach_()
            emb = self.mappinglayer(emb)
        x = self.relu(emb)
        x = self.fc2(x)
        x = self.relu(x)
        
        x1 = None
        if do_preview:
            x1 = self.preview(x, com)
        
        x = self.fc3_h0(x)

        return x, emb, x1


def resnet50(net_type=None, **kwargs):
    """Constructs a ResNet-50 based model.

    Args:
        net_type (string): model architecture (only one implemented here):
                            'MapPreview' ..... architecture from MURAUER [1]
                            
    [1] https://arxiv.org/abs/1811.09497
    """
    if net_type == 'MapPreview':
        model = ResNet_Map_PreView(Bottleneck, [5, 5, 5, 5], **kwargs)
    else:
        raise UserWarning("Required net_type not implemented.")
        
    return model
