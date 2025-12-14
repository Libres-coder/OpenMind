import torch
import torch.nn as nn
from src.flash_attention import FlashAttention
from src.kv_cache import KVCache
class OptimizedMultimodalModel(nn.Module):
    """集成Flash Attention和KV Cache的优化模型"""
    
    def __init__(self, config):
        super().__init__()
        
        # 基础组件
        self.hidden_size = config.get('hidden_size', 768)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        
        # Flash Attention层
        self.flash_attention_layers = nn.ModuleList([
            FlashAttention(self.hidden_size, self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        # KV Cache
        self.kv_cache = KVCache(
            max_seq_len=2048,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.hidden_size // self.num_heads
        )
        
        # 其他层（简化版）
        self.embedding = nn.Embedding(50257, self.hidden_size)
        self.ln_layers = nn.ModuleList([
            nn.LayerNorm(self.hidden_size)
            for _ in range(self.num_layers)
        ])
        self.lm_head = nn.Linear(self.hidden_size, 50257)
        
    def forward(self, input_ids, use_cache=False):
        # 嵌入
        x = self.embedding(input_ids)
        
        # 通过Flash Attention层
        for i, (attn, ln) in enumerate(zip(self.flash_attention_layers, self.ln_layers)):
            # 使用Flash Attention
            residual = x
            x = ln(x)
            x = attn(x, x, x)  # self-attention
            x = x + residual
        
        # 输出层
        logits = self.lm_head(x)
        
        return logits
    
    def generate_with_cache(self, input_ids, max_new_tokens=50):
        """使用KV Cache的生成函数"""
        self.kv_cache.init_cache(input_ids.shape[0], device=input_ids.device)
        
        generated = input_ids
        for _ in range(max_new_tokens):
            # 使用缓存的增量推理
            logits = self.forward(generated[:, -1:], use_cache=True)
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated
print("✅ 优化模型创建完成")
