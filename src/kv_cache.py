import torch
class KVCache:
    """KV Cache实现 - 减少重复计算"""
    
    def __init__(self, max_seq_len=2048, num_layers=12, num_heads=8, head_dim=64):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.k_cache = {}
        self.v_cache = {}
        self.current_pos = 0
        
    def init_cache(self, batch_size, device='cuda'):
        """初始化缓存"""
        for layer_idx in range(self.num_layers):
            self.k_cache[layer_idx] = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=device
            )
            self.v_cache[layer_idx] = torch.zeros(
                batch_size, self.num_heads, self.max_seq_len, self.head_dim,
                device=device
            )
        self.current_pos = 0
        
    def update(self, layer_idx, k, v, pos=None):
        """更新缓存"""
        if pos is None:
            pos = self.current_pos
            
        seq_len = k.shape[2]
        self.k_cache[layer_idx][:, :, pos:pos+seq_len] = k
        self.v_cache[layer_idx][:, :, pos:pos+seq_len] = v
        
        if pos == self.current_pos:
            self.current_pos += seq_len
            
        return self.k_cache[layer_idx][:, :, :pos+seq_len], \
               self.v_cache[layer_idx][:, :, :pos+seq_len]
    
    def get(self, layer_idx):
        """获取缓存的KV"""
        return self.k_cache[layer_idx][:, :, :self.current_pos], \
               self.v_cache[layer_idx][:, :, :self.current_pos]
    
    def clear(self):
        """清空缓存"""
        self.k_cache.clear()
        self.v_cache.clear()
        self.current_pos = 0
print("✅ KV Cache模块创建完成")
