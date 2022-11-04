from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5))
        )

        self.c5 = nn.Conv2d(in_channels=8, out_channels=60, kernel_size=5)
        self.fc6 = nn.Sequential(
            nn.Linear(in_features=60, out_features=84),
            nn.Dropout()
        )
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
    print(LeNet())