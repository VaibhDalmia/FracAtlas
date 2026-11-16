import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al., 2018):
    Uses a coarser gating signal (g) to filter low-level skip connection features (x),
    suppressing non-fracture joint spaces, soft tissue, and background noise.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: number of channels in gating signal (from deeper decoder layer)
        F_l: number of channels in skip connection (from encoder layer)
        F_int: number of intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Handle potential spatial mismatch due to padding/striding
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUp(nn.Module):
    """Upscaling then Attention Gate then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # in_channels: decoder channels (F_g)
        # out_channels: encoder skip channels (F_l)
        self.ag = AttentionGate(F_g=in_channels // 2 if not bilinear else in_channels,
                               F_l=out_channels,
                               F_int=out_channels // 2)

        self.conv = DoubleConv(in_channels if not bilinear else (in_channels + out_channels), out_channels)

    def forward(self, x1, x2):
        """
        x1: gating signal from deeper decoder layer
        x2: skip connection features from encoder layer
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Apply Attention Gate on skip connection features
        x2_att = self.ag(g=x1, x=x2)

        # Concatenate attended features with upsampled features
        x = torch.cat([x2_att, x1], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net Architecture (Oktay et al., 2018) for Medical Image Segmentation.
    Introduces attention gates to focus on sparse, subtle fracture lines while
    suppressing normal joint spaces and soft tissue boundaries.
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        f = base_features

        # Contracting Path (Encoder)
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)

        # Expansive Path with Attention Gates (Decoder)
        self.up1 = AttentionUp(f * 16, f * 8, bilinear)
        self.up2 = AttentionUp(f * 8, f * 4, bilinear)
        self.up3 = AttentionUp(f * 4, f * 2, bilinear)
        self.up4 = AttentionUp(f * 2, f, bilinear)

        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with Attention Gates
        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        logits = self.outc(d1)
        return logits
