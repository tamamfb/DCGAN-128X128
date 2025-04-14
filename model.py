import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self._block(channels_img, features_d, 4, 2, 1),    # 128 → 64
            self._block(features_d, features_d*2, 4, 2, 1),     # 64 → 32
            self._block(features_d*2, features_d*4, 4, 2, 1),   # 32 → 16
            self._block(features_d*4, features_d*8, 4, 2, 1),   # 16 → 8
            self._block(features_d*8, features_d*16, 4, 2, 1),  # 8 → 4
            nn.Conv2d(features_d*16, 1, kernel_size=4, stride=1, padding=0),  # 4 → 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride, 
                    padding, 
                    bias=False
                )
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g*16, 4, 1, 0),         # 1 → 4
            self._block(features_g*16, features_g*8, 4, 2, 1),   # 4 → 8
            self._block(features_g*8, features_g*4, 4, 2, 1),    # 8 → 16
            self._block(features_g*4, features_g*2, 4, 2, 1),    # 16 → 32
            self._block(features_g*2, features_g, 4, 2, 1),      # 32 → 64
            nn.ConvTranspose2d(
                features_g, 
                channels_img, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),   # 64 → 128
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)