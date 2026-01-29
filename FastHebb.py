from torch import nn
from HebbConv2d import HebbConv2d

class FastHebb(nn.Module):
    def __init__(self):
        super(FastHebb, self).__init__()

        # block 1
        self.conv1 = HebbConv2d(in_channels=3, out_channels=96, kernel_size=5, padding=0)
        self.activ1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(96, affine=False) # 14*14

        # block 2
        self.conv2 = HebbConv2d(in_channels=96, out_channels=128, kernel_size=3, padding=0)
        self.activ2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128, affine=False) # 12*12

        # block 3
        self.conv3 = HebbConv2d(in_channels=128, out_channels=192, kernel_size=3, padding=0)
        self.activ3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(192, affine=False) # 5*5

        # block 4
        self.conv4 = HebbConv2d(in_channels=192, out_channels=256, kernel_size=3, padding=0)
        self.activ4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256, affine=False) # 3*3

        # block 5 FC 4096
        self.fc_in_channels = 256 * 3 * 3  # 256 channels, each 3x3 after conv4 and pool3
        self.fc5 = nn.Linear(in_features=self.fc_in_channels, out_features=4096, bias=True)
        self.activ5 = nn.ReLU()
        self.bn5 = nn.BatchNorm1d(4096)
        self.dropout = nn.Dropout(0.5)

        # block 6 Classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(4096, 10)

    def forward(self, x):
        # block 1
        out = self.conv1(x)
        out = self.activ1(out)
        out = self.pool1(out)
        out = self.bn1(out)
        # block 2
        out = self.conv2(out)
        out = self.activ2(out)
        out = self.bn2(out)
        # block 3
        out = self.conv3(out)
        out = self.activ3(out)
        out = self.pool3(out)
        out = self.bn3(out)
        # block 4
        out = self.conv4(out)
        out = self.activ4(out)
        out = self.bn4(out)
        # block 5
        out = self.flatten(out)
        out = self.fc5(out)
        out = self.activ5(out)
        out = self.bn5(out)
        out = self.dropout(out)
        # block 6 Classifier
        out = self.classifier(out)
        return out
