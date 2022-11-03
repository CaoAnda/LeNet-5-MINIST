from torch import nn


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.s2 = nn.AdaptiveAvgPool2d((14, 14))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AdaptiveAvgPool2d((5, 5))
        
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc6 = nn.Linear(in_features=120, out_features=84)
        self.fc7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        out = nn.ReLU(self.c1(x))
        out = nn.ReLU(self.s2(out))
        out = nn.ReLU(self.c3(out))
        out = nn.ReLU(self.s4(out))
        out = nn.ReLU(self.c5(out))
        out = nn.ReLU(self.fc6(out))
        out = nn.ReLU(self.fc7(out))

        return out