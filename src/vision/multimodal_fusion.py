import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
class CrossModalAttention(nn.Module):
    """跨模态注意力 - 让文本关注图像，图像关注文本"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.shape
        kv_len = key_value.shape[1]
        
        # 投影
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim)
        
        # 转置
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        
        # 输出
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(out)
class MultimodalFusion(nn.Module):
    """多模态融合模块 - 融合视觉和文本特征"""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 8,
                 num_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 视觉->文本 注意力
        self.vision_to_text = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # 文本->视觉 注意力
        self.text_to_vision = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # 自注意力
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # FFN
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers * 4)
        ])
        
        # 融合输出
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, text_features: torch.Tensor,
                vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        融合文本和视觉特征
        
        Args:
            text_features: [batch, text_len, embed_dim]
            vision_features: [batch, vision_len, embed_dim]
        """
        text_out = text_features
        vision_out = vision_features
        
        for i in range(len(self.vision_to_text)):
            # 跨模态注意力
            text_cross = self.vision_to_text[i](text_out, vision_out)
            text_out = self.norms[i*4](text_out + text_cross)
            
            vision_cross = self.text_to_vision[i](vision_out, text_out)
            vision_out = self.norms[i*4+1](vision_out + vision_cross)
            
            # 自注意力
            text_self, _ = self.self_attn[i](text_out, text_out, text_out)
            text_out = self.norms[i*4+2](text_out + text_self)
            
            # FFN
            text_out = self.norms[i*4+3](text_out + self.ffn[i](text_out))
        
        # 融合门控
        text_pool = text_out.mean(dim=1)  # [batch, embed_dim]
        vision_pool = vision_out.mean(dim=1)
        
        gate = self.fusion_gate(torch.cat([text_pool, vision_pool], dim=-1))
        fused = gate * text_pool + (1 - gate) * vision_pool
        
        return {
            'text_features': text_out,
            'vision_features': vision_out,
            'fused_features': fused,
            'gate_values': gate
        }
print("✅ 多模态融合模块创建完成")
