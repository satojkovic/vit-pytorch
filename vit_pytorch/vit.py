import torch
import torch.nn as nn


class Patches(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, patch_dim=192):
        super(Patches, self).__init__()
        self.in_channels = in_channels
        self.patch_dim = patch_dim
        self.patch_size = patch_size

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.patch_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, D, H/P, W/P)
        patches = self.conv(x)
        b, d, h, w = patches.shape

        # (B, D, H/P, W/P) -> (B, D, H*W/P^2)
        # Np = H*W/P^2
        patches = patches.view(b, d, -1)

        # (B, D, Np) -> (B, Np, D)
        patches = patches.transpose(1, 2)

        return patches


if __name__ == "__main__":
    patch_extractor = Patches()
    dummy_x = torch.randn((10, 3, 224, 224))
    patches = patch_extractor(dummy_x)
    print(f"patches: {patches.shape}")
