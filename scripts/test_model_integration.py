"""测试改进组件集成到主模型"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from model_architecture import MultimodalReasoningModel

def test_model_with_improved_components():
    """测试使用改进组件的模型"""
    print("="*60)
    print("测试改进组件集成")
    print("="*60)
    
    # 配置 - 使用较小的模型进行测试
    config = {
        'base_model': 'Qwen/Qwen2-0.5B',  # 使用小模型快速测试
        'img_size': 384,
        'patch_size': 14,
        'vision_embed_dim': 1152,
        'vision_depth': 27,
        'vision_heads': 16,
        'use_flash_attn': True,
        'projector_type': 'token_pooling',
        'pooling_kernel': 2,
        'enable_audio': False,
        'enable_cot': True,
        'enable_verification': True
    }
    
    try:
        print("\n[1/4] 创建模型...")
        model = MultimodalReasoningModel(config)
        print("✅ 模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n[2/4] 模型参数统计:")
        print(f"  总参数: {total_params / 1e9:.2f}B")
        print(f"  可训练参数: {trainable_params / 1e9:.2f}B")
        print(f"  可训练比例: {trainable_params / total_params * 100:.1f}%")
        
        # 测试前向传播
        print(f"\n[3/4] 测试前向传播...")
        batch_size = 2
        seq_len = 32
        
        # 创建测试数据（转换为bfloat16）
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        images = torch.randn(batch_size, 3, 384, 384).to(torch.bfloat16)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # 前向传播（不使用labels）
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=images
            )
        
        print("✅ 前向传播成功")
        print(f"  输出logits shape: {outputs['logits'].shape}")
        print(f"  Hidden states shape: {outputs['hidden_states'].shape}")
        
        # 测试生成
        print(f"\n[4/4] 测试生成功能...")
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids[:1],
                images=images[:1],
                max_length=50,
                temperature=0.7
            )
        
        print("✅ 生成功能正常")
        print(f"  生成的token数: {generated.shape[1]}")
        
        print("\n" + "="*60)
        print("🎉 所有测试通过！改进组件集成成功！")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model_with_improved_components()
    sys.exit(0 if success else 1)
