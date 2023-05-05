import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("models/")
from vgg16 import vgg16_bn


def make_layers():
    _vgg16_bn = vgg16_bn(32, pretrained=False)
    features = list(_vgg16_bn.features.children())
    classifier = list(_vgg16_bn.classifier.children())
    
    conv1 = nn.Sequential(*features[:6])
    conv2 = nn.Sequential(*features[7:13])
    conv3 = nn.Sequential(*features[14:23])
    conv4 = nn.Sequential(*features[24:33])
    conv5 = nn.Sequential(*features[34:43])
    
    conv6 = nn.Conv2d(512, 4096, kernel_size=(7, 7))
    conv7 = nn.Conv2d(4096, 4096, kernel_size=(1, 1))
    
    w_conv6 = classifier[0].state_dict()
    w_conv7 = classifier[3].state_dict()
    
    conv6.load_state_dict({'weight':w_conv6['weight'].view(4096, 512, 7, 7), 'bias':w_conv6['bias']})
    conv7.load_state_dict({'weight':w_conv7['weight'].view(4096, 4096, 1, 1), 'bias':w_conv7['bias']})

    return [conv1, conv2, conv3, conv4, conv5, conv6, conv7]