import torch.nn as nn
from torchsummary import summary

import sys
sys.path.append("models/")
from make_layers_encoder import make_layers
from attention import PAM_CAM_Layer


class DeconvNetv2(nn.Module):
    def __init__(self, num_classes, init_weights, is_sk, is_attent):
        """Define the DeconvNetv2

        Args:
            num_classes (int): number of classes in dataset
            init_weights (bool): initial weight for layers
            is_sk (bool): using skip connection or not
            is_attent (bool): using attention or not
        """
        super(DeconvNetv2, self).__init__()

        self.is_sk = is_sk
        self.is_attent = is_attent
        
        layers = make_layers()
        
        # 224 x 224
        self.conv1 = layers[0]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        # 112 x 112
        self.conv2 = layers[1]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        # 56 x 56
        self.conv3 = layers[2]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        # 28 x 28
        self.conv4 = layers[3]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        # 14 x 14
        self.conv5 = layers[4]
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        # 1 x 1
        self.conv67 = nn.Sequential(layers[5], nn.BatchNorm2d(4096), nn.ReLU(),
                                    layers[6], nn.BatchNorm2d(4096), nn.ReLU())
        
        # 7 x 7
        self.deconv67 = nn.Sequential(nn.ConvTranspose2d(4096, 512, kernel_size=7, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU())
        
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.attention_block_5 = PAM_CAM_Layer(512)
        # 14 x 14
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.attention_block_4 = PAM_CAM_Layer(512)
        # 28 x 28
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        self.deconv4_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.attention_block_3 = PAM_CAM_Layer(256)
        # 56 x 56
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.deconv3_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.attention_block_2 = PAM_CAM_Layer(128)
        # 112 x 112
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv2_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.attention_block_1 = PAM_CAM_Layer(64)
        # 224 x 224
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.deconv1_2 = nn.Sequential(
            nn.ConvTranspose2d(64, num_classes, kernel_size=1, stride=1, padding=0))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        original = x
        
        x = self.conv1(x)
        if self.is_sk:
            x1 = x
        x, p1 = self.pool1(x)
        
        x = self.conv2(x)
        if self.is_sk:
            x2 = x
        x, p2 = self.pool2(x)
        
        x = self.conv3(x)
        if self.is_sk:
            x3 = x
        x, p3 = self.pool3(x)
        
        x = self.conv4(x)
        if self.is_sk:
            x4 = x
        x, p4 = self.pool4(x)
        
        x = self.conv5(x)
        if self.is_sk:
            x5 = x
        x, p5 = self.pool5(x)
        
        
        x = self.conv67(x)
        x = self.deconv67(x)
        
        x = self.unpool5(x, p5)
        if self.is_sk:
            x += x5
        if self.is_attent:
            attention_5 = self.attention_block_5(x)
            x = self.deconv5(x) + attention_5
        else:
            x = self.deconv5(x)
        
        x = self.unpool4(x, p4)
        if self.is_sk:
            x += x4
        if self.is_attent:
            attention_4 = self.attention_block_4(x)
            x = self.deconv4(x) + attention_4
            x = self.deconv4_2(x)
        else:
            x = self.deconv4(x)
            x = self.deconv4_2(x)
        
        x = self.unpool3(x, p3)
        if self.is_sk:
            x += x3
        if self.is_attent:
            attention_3 = self.attention_block_3(x)
            x = self.deconv3(x) + attention_3
            x = self.deconv3_2(x)
        else:
            x = self.deconv3(x)
            x = self.deconv3_2(x)
        
        x = self.unpool2(x, p2)
        if self.is_sk:
            x += x2
        if self.is_attent:
            attention_2 = self.attention_block_2(x)
            x = self.deconv2(x) + attention_2
            x = self.deconv2_2(x)
        else:
            x = self.deconv2(x)
            x = self.deconv2_2(x)
        
        x = self.unpool1(x, p1)
        if self.is_sk:
            x += x1
        if self.is_attent:
            attention_1 = self.attention_block_1(x)
            x = self.deconv1(x) + attention_1
            x = self.deconv1_2(x)
        else:
            x = self.deconv1(x)
            x = self.deconv1_2(x)
        
        return x

    def _initialize_weights(self):
        targets = [self.conv67, self.deconv67, self.deconv5, self.deconv4, self.deconv3, self.deconv2, self.deconv1]
        for layer in targets:
            for module in layer:
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


if __name__ == '__main__':
    model = DeconvNetv2(num_classes=21, init_weights=True, is_sk=True, is_attent=True)
    summary(model, (3, 224, 224))