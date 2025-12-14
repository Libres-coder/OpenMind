import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class FlashAttention(nn.Module):
    """Flash Attention实现 - 优化内存使用和计算速度"""
    
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
    def forward(self, q, k, v, mask=None):
        B, N, C = q.shape
        
        # 重塑为多头
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用PyTorch 2.0的scaled_dot_product_attention
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )
        else:
            # 标准注意力计算作为后备
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn, v)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        
        return attn_output
print("✅ Flash Attention模块创建完成")
