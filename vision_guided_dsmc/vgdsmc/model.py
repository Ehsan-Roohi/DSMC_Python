from __future__ import annotations


def build_unet(in_channels: int = 4, classes: int = 3):
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies: pip install -e '.[ml]'") from exc

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = Block(in_channels, 16)
            self.e2 = Block(16, 32)
            self.bottleneck = Block(32, 64)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, 2)
            self.d2 = Block(64, 32)
            self.d1 = Block(32, 16)
            self.out = nn.Conv2d(16, classes, 1)

        @staticmethod
        def _match(x, reference):
            if x.shape[-2:] != reference.shape[-2:]:
                x = functional.interpolate(
                    x, size=reference.shape[-2:], mode="bilinear", align_corners=False
                )
            return x

        def forward(self, x):
            a = self.e1(x)
            b = self.e2(self.pool(a))
            z = self.bottleneck(self.pool(b))
            z = self._match(self.up2(z), b)
            z = self.d2(torch.cat([z, b], dim=1))
            z = self._match(self.up1(z), a)
            z = self.d1(torch.cat([z, a], dim=1))
            return self.out(z)

    return SmallUNet()
