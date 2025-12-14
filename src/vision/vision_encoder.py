import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
class PatchEmbedding(nn.Module):
    """将图像分割为patches并嵌入"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, height, width]
        x = self.proj(x)  # [batch, embed_dim, h/p, w/p]
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        return x
class VisionEncoder(nn.Module):
    """视觉编码器 - 将图像编码为特征向量"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768,
                 num_layers: int = 6, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch嵌入
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        
        # Patch嵌入
        x = self.patch_embed(x)  # [batch, num_patches, embed_dim]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_patches+1, embed_dim]
        
        # 添加位置嵌入
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)
        x = self.norm(x)
        
        return {
            'cls_token': x[:, 0],  # [batch, embed_dim]
            'patch_tokens': x[:, 1:],  # [batch, num_patches, embed_dim]
            'all_tokens': x  # [batch, num_patches+1, embed_dim]
        }
print("✅ 视觉编码器创建完成")
