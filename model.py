import torch
import torch.nn as nn
import torchvision.transforms.functional as TTF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.Conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.Ascend = nn.ModuleList()
        self.Descend = nn.ModuleList()
        self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #UNET Encoder
        for feature in features:
            self.Descend.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #UNET Decoder
        for feature in reversed(features):
            self.Ascend.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.Ascend.append(DoubleConv(feature*2, feature))

        self.Vertex = DoubleConv(features[-1], features[-1]*2)
        self.Supremum = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for Descend in self.Descend:
            x = Descend(x)
            skip_connections.append(x)
            x = self.Pool(x)

        x = self.Vertex(x)
        skip_connections = skip_connections[::-1]

        for index in range(0, len(self.Ascend), 2):
            x = self.Ascend[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape:
                x = TTF.resize(x, size=skip_connection.shape[2:])

            concatenate = torch.cat((skip_connection, x), dim=1)
            x = self.Ascend[index+1](concatenate)

        return self.Supremum(x)