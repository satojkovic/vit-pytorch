import torch
import torch.nn as nn
import einops


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


class PatchEncoder(nn.Module):
    def __init__(self, num_patches, patch_dim=192, embed_dim=192):
        super(PatchEncoder, self).__init__()
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim
        self.encoder = nn.Linear(
            in_features=self.patch_dim, out_features=self.embed_dim
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, self.embed_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        # Add cls token
        cls_tokens = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add position embedding
        x += self.pos_embedding

        return x


if __name__ == "__main__":
    patch_extractor = Patches()
    dummy_x = torch.randn((10, 3, 224, 224))
    patches = patch_extractor(dummy_x)
    print(f"patches: {patches.shape}")

    num_patches = patches.shape[1]
    patch_encoder = PatchEncoder(num_patches=num_patches)
    encoded_patches = patch_encoder(patches)
    print(f"encoded_patches: {encoded_patches.shape}")
