

import torch
import torch.nn as nn
import torch.nn.functional as F
import params
from torch.nn.utils import spectral_norm



class net_G(nn.Module):
    def __init__(self, args=None):
        super(net_G, self).__init__()
        self.z_dim = params.z_dim
        self.cond_dim = getattr(params, "cond_dim", 0)
        self.in_dim = self.z_dim + self.cond_dim

        self.fc = nn.Linear(self.in_dim, 512*2*2*2)

        self.layer1 = self.deconv_block(512, 256)
        self.layer2 = self.deconv_block(256, 128)
        self.layer3 = self.deconv_block(128, 64)
        self.layer4 = self.deconv_block(64, 32)
        self.layer5 = self.deconv_block(32, 16)

        self.out = nn.ConvTranspose3d(16, 1, 3, stride=1, padding=1)

    def deconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, z, y=None):
        if y is not None:
            z = torch.cat([z, y], dim=1)

        B = z.size(0)
        h = self.fc(z)
        h = h.view(B, 512, 2, 2, 2)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)

        return torch.tanh(self.out(h))



class net_D(nn.Module):
    def __init__(self, args=None):
        super(net_D, self).__init__()
        self.cube_len  = params.cube_len
        self.cond_dim  = getattr(params, "cond_dim", 0)

        # Base CNN 
        self.layer1 = self.conv_block(1,   32, normalize=False)
        self.layer2 = self.conv_block(32,  64)
        self.layer3 = self.conv_block(64, 128)
        self.layer4 = self.conv_block(128, 256)

        # Global pooling
        self.final_pool = nn.AdaptiveAvgPool3d(1)

        # Output head
        self.fc = nn.Linear(256, 1)

        # Small embedding for y (projection discriminator)
        if self.cond_dim > 0:
            self.embed_y = nn.Linear(self.cond_dim, 256)
        else:
            self.embed_y = None

    def conv_block(self, in_ch, out_ch, normalize=True):
        layers = [
            spectral_norm(
                nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if normalize:
            layers.insert(1, nn.BatchNorm3d(out_ch))
        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        if x.dim() == 4:
            x = x.unsqueeze(1)

        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = self.final_pool(h).view(h.size(0), -1)
        out = self.fc(h)

        # projection term (conditional part)
        if self.embed_y is not None and y is not None:
            out = out + torch.sum(self.embed_y(y) * h, dim=1, keepdim=True)

        return out
