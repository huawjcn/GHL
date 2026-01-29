from torch import nn
from HebbConv2d import HebbConv2d


class DeepHebb(nn.Module):
    def __init__(self):
        super(DeepHebb, self).__init__()

        # block 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbConv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2, t_invert=1.0)
        self.activ1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        # block 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbConv2d(in_channels=96, out_channels=384, kernel_size=3, padding=1, t_invert=0.65)
        self.activ2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = HebbConv2d(in_channels=384, out_channels=1536, kernel_size=3, padding=1, t_invert=0.25)
        self.activ3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # block 4 - Classifier
        self.flatten = nn.Flatten()
        # -> 32x32 -> conv1(pad2,k5,s1)->32x32 -> pool1(k4,s2,p1)-> 16x16
        # -> conv2(pad1,k3,s1)->16x16 -> pool2(k4,s2,p1)-> 8x8
        # -> conv3(pad1,k3,s1)->8x8 -> pool3(k2,s2,p0)-> 4x4
        self.classifier_input_features = 1536 * 4 * 4
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.classifier_input_features, 10)

    def forward(self, x):
        # block 1
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.activ1(out)
        out = self.pool1(out)
        # block 2
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.activ2(out)
        out = self.pool2(out)
        # block 3
        out = self.bn3(out)
        out = self.conv3(out)
        out = self.activ3(out)
        out = self.pool3(out)
        # block 4
        out = self.flatten(out)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
