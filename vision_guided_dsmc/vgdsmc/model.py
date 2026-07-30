from __future__ import annotations


def build_unet(in_channels: int = 4, classes: int = 3):
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies: pip install -e '.[ml]'") from exc

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())
        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.b = Block(in_channels, 16), Block(16, 32), Block(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.up2, self.up1 = nn.ConvTranspose2d(64, 32, 2, 2), nn.ConvTranspose2d(32, 16, 2, 2)
            self.d2, self.d1 = Block(64, 32), Block(32, 16)
            self.out = nn.Conv2d(16, classes, 1)
        def forward(self, x):
            a = self.e1(x)
            b = self.e2(self.pool(a))
            z = self.b(self.pool(b))
            z = self.d2(torch.cat([self.up2(z), b], dim=1))
            z = self.d1(torch.cat([self.up1(z), a], dim=1))
            return self.out(z)

    return SmallUNet()
