import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """Patch Embedding with Conv2D"""
    def __init__(self, img_size=600, patch_size=16, in_chans=28, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Convert to patch embeddings
        return x


class ViTBackbone(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, img_size=600, patch_size=16):
        super(ViTBackbone, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2  # 计算 patch 数量
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=28, embed_dim=embed_dim)  # 初始化 patch embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))  # 位置嵌入

        # 初始化 transformer 模块
        self.transformer = timm.models.vision_transformer.VisionTransformer(
            patch_size=1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0  # 去掉分类头
        )
        self.norm = nn.LayerNorm(embed_dim)  # 最后的层归一化

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add position embedding
        x = x + self.pos_embed

        # Transformer forward
        for block in self.transformer.blocks:
            x = block(x)  # 显式调用每个 transformer block

        x = self.norm(x)  # [B, num_patches, embed_dim]

        # Reshape back to 2D grid
        x = x.transpose(1, 2).view(B, C, H, W)  # [B, embed_dim, H/patch_size, W/patch_size]
        return x


class Decoder(nn.Module):
    """Decoder to upsample transformer features to target size"""
    def __init__(self, input_dim, output_dim, target_size=(600, 600)):
        super(Decoder, self).__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output normalized between 0 and 1
        )

    def forward(self, x):
        x = self.upsample(x)
        #将维度从1,300,592,592转换为1,300,600,600
        x = F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=False)

        return x

class VIT_base(nn.Module):
    """Vision Transformer for Light Field Reconstruction"""
    def __init__(self):
        super(VIT_base, self).__init__()
        self.backbone = ViTBackbone(embed_dim=768, depth=12, num_heads=12, img_size=600, patch_size=16)
        self.decoder = Decoder(input_dim=768, output_dim=300, target_size=(600, 600))

    def forward(self, x):
        # ViT backbone
        x = self.backbone(x)

        # Decoder to reconstruct 3D light field
        x = self.decoder(x)

        return x

# 测试模型
if __name__ == "__main__":
    # 初始化模型
    model = VIT_base()
    #print(model)

    # 测试输入和输出
    input_tensor = torch.randn(1, 28, 600, 600)  # [B, C, H, W]
    output = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
