import torch.nn as nn
# import torchvision.models as models

import sys
sys.path.append("models/")
from vgg16 import VGG16


def make_layers():
    """make encoder layers of DeconvNet (using VGG16 architecture)

    Returns:
        list: list of conv layers
    """
    # vgg16_bn = models.vgg16_bn(pretrained=True)
    vgg16_bn = VGG16(num_classes=1000)
    features = list(vgg16_bn.features.children())
    classifier = list(vgg16_bn.classifier.children())
    
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