# app/model.py
# Keep this in sync with the SiameseUNet class in 02_train.ipynb

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for satellite damage detection.
    Shared encoder + absolute-difference skip connections + U-Net decoder.
    Output: (B, num_classes, H, W) logits.
    """
    def __init__(self, in_ch=3, num_classes=5, features=[32, 64, 128, 256]):
        super().__init__()
        f = features
        self.enc1 = DoubleConv(in_ch, f[0])
        self.enc2 = DoubleConv(f[0],  f[1])
        self.enc3 = DoubleConv(f[1],  f[2])
        self.enc4 = DoubleConv(f[2],  f[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f[3], f[3]*2)
        self.up4  = nn.ConvTranspose2d(f[3]*2, f[3], 2, stride=2)
        self.dec4 = DoubleConv(f[3]*2, f[3])
        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2]*2, f[2])
        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1]*2, f[1])
        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0]*2, f[0])
        self.out_conv = nn.Conv2d(f[0], num_classes, 1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def forward(self, pre, post):
        pre_e1,  pre_e2,  pre_e3,  pre_e4,  pre_b  = self.encode(pre)
        post_e1, post_e2, post_e3, post_e4, post_b = self.encode(post)
        b  = torch.abs(post_b  - pre_b)
        d4 = self.dec4(torch.cat([self.up4(b),  torch.abs(post_e4-pre_e4)], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), torch.abs(post_e3-pre_e3)], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), torch.abs(post_e2-pre_e2)], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), torch.abs(post_e1-pre_e1)], 1))
        return self.out_conv(d1)
