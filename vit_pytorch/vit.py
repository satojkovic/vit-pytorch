import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
import math


class Patches(nn.Module):
    def __init__(self, image_size=224, in_channels=3, patch_size=16):
        super(Patches, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        assert self.image_size % self.patch_size == 0
        self.patch_dim = self.in_channels * (self.patch_size**2)

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

        # (B, D, H/P, W/P) -> (B, D, Np)
        # Np = H*W/P^2
        patches = patches.view(b, d, -1)

        # (B, D, Np) -> (B, Np, D)
        patches = patches.transpose(1, 2)

        return patches


class PatchEncoder(nn.Module):
    def __init__(self, num_patches, patch_dim, embed_dim):
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, drop_p):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.q_net = nn.Linear(embed_dim, embed_dim)
        self.k_net = nn.Linear(embed_dim, embed_dim)
        self.v_net = nn.Linear(embed_dim, embed_dim)
        self.proj_net = nn.Linear(embed_dim, embed_dim)  # W_o

        self.attn_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

    def forward(self, x):
        # B: batch size, T: sequence length, D: embedding dimension
        B, T, D = x.shape

        k = self.num_heads
        Dh = D // k

        q = self.q_net(x).reshape(B, T, k, Dh).transpose(0, 2, 1, 3)  # (B, k, T, Dh)
        k = self.k_net(x).reshape(B, T, k, Dh).transpose(0, 2, 1, 3)
        v = self.v_net(x).reshape(B, T, k, Dh).transpose(0, 2, 1, 3)

        # attention matrix
        weights = q @ k.transpose(2, 3) / math.sqrt(Dh)  # (B, k, T, T)
        normalized_weights = F.softmax(weights, dim=-1)

        # attention
        attention = self.attn_drop(normalized_weights @ v)  # (B, k, T, Dh)

        # gather head
        attention = attention.transpose(1, 2).view(B, T, k * Dh)

        out = self.proj_drop(self.proj_net(attention))
        return out


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout_ratio=0.5):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout_ratio),
        )

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, embed_dim, mlp_dim, drop_p):
        super().__init__()
        self.layer_norm_mha = nn.LayerNorm(embed_dim)
        self.mha = MultiHeadSelfAttention(num_heads, embed_dim, drop_p)
        self.layer_norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_dim, drop_p)

    def forward(self, x):
        x = self.mha(self.layer_norm_mha(x)) + x
        out = self.mlp(self.layer_norm_mlp(x)) + x
        return out


if __name__ == "__main__":
    num_batches = 10
    image_size, in_channels = 224, 3
    patch_size = 16

    patch_extractor = Patches(image_size, in_channels, patch_size)
    dummy_x = torch.randn((num_batches, in_channels, image_size, image_size))
    patches = patch_extractor(dummy_x)
    print(f"patches: {patches.shape}")

    num_patches = patches.shape[1]
    patch_dim = patches.shape[2]
    embed_dim = 768
    patch_encoder = PatchEncoder(num_patches, patch_dim, embed_dim)
    encoded_patches = patch_encoder(patches)
    print(f"encoded_patches: {encoded_patches.shape}")
