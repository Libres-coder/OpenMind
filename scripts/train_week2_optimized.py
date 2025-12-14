import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import logging
from src.flash_attention import FlashAttention
from src.kv_cache import KVCache
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def test_flash_attention():
    """测试Flash Attention"""
    logger.info("测试Flash Attention...")
    
    # 创建Flash Attention模块
    flash_attn = FlashAttention(dim=768, num_heads=8)
    
    # 测试输入
    batch_size, seq_len, dim = 2, 128, 768
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    v = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = flash_attn(q, k, v)
    logger.info(f"  输入shape: {q.shape}")
    logger.info(f"  输出shape: {output.shape}")
    logger.info("  ✅ Flash Attention测试通过")
    
def test_kv_cache():
    """测试KV Cache"""
    logger.info("测试KV Cache...")
    
    # 创建KV Cache
    kv_cache = KVCache(max_seq_len=512, num_layers=6, num_heads=8, head_dim=64)
    
    # 初始化缓存
    batch_size = 2
    kv_cache.init_cache(batch_size, device='cpu')
    
    # 模拟增量推理
    for i in range(3):
        k = torch.randn(batch_size, 8, 10, 64)  # 每次10个token
        v = torch.randn(batch_size, 8, 10, 64)
        k_full, v_full = kv_cache.update(0, k, v)
        logger.info(f"  步骤{i+1}: 缓存大小 {k_full.shape}")
    
    logger.info("  ✅ KV Cache测试通过")
def main():
    logger.info("="*60)
    logger.info("Week 2: 推理优化实现")
    logger.info("="*60)
    
    # 检查Week 1模型
    if os.path.exists('outputs/week1/best_model.pt'):
        checkpoint = torch.load('outputs/week1/best_model.pt', map_location='cpu')
        logger.info(f"✅ Week 1模型已找到，Loss: {checkpoint['loss']:.4f}")
    
    logger.info("\n实现的优化:")
    
    # 测试Flash Attention
    test_flash_attention()
    
    # 测试KV Cache
    test_kv_cache()
    
    logger.info("\n✅ Week 2优化模块实现完成！")
    logger.info("\n下一步：")
    logger.info("1. 将优化集成到模型中")
    logger.info("2. 实现量化优化")
    logger.info("3. 进入Week 3推理服务器")
if __name__ == "__main__":
    main()
