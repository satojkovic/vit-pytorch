import torch
import torch.nn as nn
from torch.nn import functional as F
import einops
import math


class PatchEncoder(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, embed_dim):
        super(PatchEncoder, self).__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.patch_dim = in_channels * (patch_size**2)
        self.num_patches = self.image_size**2 // self.patch_size**2

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.patch_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.embed_dim = embed_dim
        self.encoder = nn.Linear(
            in_features=self.patch_dim, out_features=self.embed_dim
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim)
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, D, H/P, W/P)
        patches = self.conv(x)
        b, d, _, _ = patches.shape

        # (B, D, H/P, W/P) -> (B, D, Np)
        # Np = H*W/P^2
        patches = patches.view(b, d, -1)

        # (B, D, Np) -> (B, Np, D)
        patches = patches.transpose(1, 2)

        z = self.encoder(patches)

        # Add cls token
        cls_tokens = einops.repeat(self.cls_token, "1 1 d -> b 1 d", b=b)

        z = torch.cat((cls_tokens, z), dim=1)

        # Add position embedding
        z += self.pos_embedding

        return z


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

        Dh = D // self.num_heads

        q = (
            self.q_net(x).view(B, T, self.num_heads, Dh).transpose(1, 2)
        )  # (B, k, T, Dh)
        k = self.k_net(x).view(B, T, self.num_heads, Dh).transpose(1, 2)
        v = self.v_net(x).view(B, T, self.num_heads, Dh).transpose(1, 2)

        # attention matrix
        weights = q @ k.transpose(2, 3) / math.sqrt(Dh)  # (B, k, T, T)
        normalized_weights = F.softmax(weights, dim=-1)

        # attention
        attention = self.attn_drop(normalized_weights @ v)  # (B, k, T, Dh)

        # gather head
        attention = (
            attention.transpose(1, 2).contiguous().view(B, T, self.num_heads * Dh)
        )

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


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        patch_size,
        num_layers,
        embed_dim,
        mlp_dim,
        num_heads,
        drop_p,
        num_classes,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.patch_encoder = PatchEncoder(
            self.image_size, self.in_channels, self.patch_size, self.embed_dim
        )

        self.transformer_encoder = TransformerEncoder(
            self.num_heads, self.embed_dim, self.mlp_dim, self.drop_p
        )

        self.cls_head = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        z = self.patch_encoder(x)
        y = self.transformer_encoder(z)
        out = self.cls_head(y)
        return out


if __name__ == "__main__":
    import torchsummary

    num_batches = 10
    image_size, in_channels = 224, 3
    patch_size = 16
    embed_dim = 768
    num_layers = 12
    mlp_dim = 3072
    num_heads = 12
    drop_p = 0.5
    num_classes = 10

    dummy_x = torch.randn((num_batches, in_channels, image_size, image_size))
    vit_model = ViT(
        image_size=image_size,
        in_channels=in_channels,
        patch_size=patch_size,
        num_layers=num_layers,
        embed_dim=embed_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        drop_p=drop_p,
        num_classes=num_classes,
    )
    out = vit_model(dummy_x)
    print(f"out: {out.shape}")
    torchsummary.summary(vit_model, input_size=(in_channels, image_size, image_size))
