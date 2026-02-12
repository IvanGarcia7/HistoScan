import torch
import torch.nn as nn

class UNetLite(nn.Module):
    def __init__(self, in_channels=3, base=32):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base*2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.dec1 = nn.Sequential(
            nn.Conv2d(base*3, base, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        d1 = self.up(e2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


