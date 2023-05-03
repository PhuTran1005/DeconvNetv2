import torch.nn as nn
from torchsummary import summary

from make_layers_encoder import make_layers


class DeconvNetv2(nn.Module):
    def __init__(self, num_classes, init_weights):
        """Define the DeconvNetv2

        Args:
            num_classes (int): number of classes in dataset
            init_weights (bool): initial weight for layers
        """
        super(DeconvNetv2, self).__init__()
        
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
        # 14 x 14
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU())
        
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 28 x 28
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 56 x 56
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 112 x 112
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 224 x 224
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=1, stride=1, padding=0))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        original = x
        
        x = self.conv1(x)
        x, p1 = self.pool1(x)
        
        x = self.conv2(x)
        x, p2 = self.pool2(x)
        
        x = self.conv3(x)
        x, p3 = self.pool3(x)
        
        x = self.conv4(x)
        x, p4 = self.pool4(x)
        
        x = self.conv5(x)
        x, p5 = self.pool5(x)
        
        
        x = self.conv67(x)
        x = self.deconv67(x)
        
        x = self.unpool5(x, p5)
        x = self.deconv5(x)
        
        x = self.unpool4(x, p4)
        x = self.deconv4(x)
        
        x = self.unpool3(x, p3)
        x = self.deconv3(x)
        
        x = self.unpool2(x, p2)
        x = self.deconv2(x)
        
        x = self.unpool1(x, p1)
        x = self.deconv1(x)
        
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
    model = DeconvNetv2(num_classes=21, init_weights=True)
    summary(model, (3, 224, 224))