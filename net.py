from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self, half_conv, dropout) -> None:
        super().__init__()

        half = 2 if half_conv else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6//half, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6//half, out_channels=16//half, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5))
        )

        self.c5 = nn.Conv2d(in_channels=16//half, out_channels=120//half, kernel_size=5)
        self.fc6 = nn.Sequential(
            nn.Linear(in_features=120//half, out_features=84),
        )
        if dropout != 0:
            self.fc6.append(nn.Dropout(dropout))
        self.fc7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(self.c5(out))
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc6(out))
        out = self.fc7(out)

        return out

if __name__ == '__main__':
    print(LeNet(half_conv=False, dropout=0))