"""
测试模型架构
验证模型可以正常创建和前向传播
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model_architecture import create_model

def test_model_creation():
    print("="*60)
    print("🧪 测试模型架构")
    print("="*60)
    
    print("\n1️⃣ 创建模型配置...")
    config = {
        'base_model': 'Qwen/Qwen2-7B',
        'vision_model': 'openai/clip-vit-large-patch14',
        'freeze_vision': True,
        'perceiver_depth': 2,
        'num_latents': 32,
        'enable_audio': False,
        'enable_cot': True,
        'enable_verification': True
    }
    
    print("配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n2️⃣ 创建模型...")
    try:
        model = create_model(config)
        print("✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3️⃣ 分析模型参数...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"  总参数: {total_params/1e9:.2f}B ({total_params:,})")
    print(f"  可训练参数: {trainable_params/1e9:.2f}B ({trainable_params:,})")
    print(f"  冻结参数: {frozen_params/1e9:.2f}B ({frozen_params:,})")
    print(f"  可训练比例: {trainable_params/total_params*100:.2f}%")
    
    print("\n4️⃣ 测试前向传播...")
    batch_size = 2
    seq_len = 128
    
    try:
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        
        print(f"  输入shape: {input_ids.shape}")
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        print(f"✅ 前向传播成功")
        print(f"  输出logits shape: {outputs['logits'].shape}")
        if outputs['loss'] is not None:
            print(f"  Loss: {outputs['loss'].item():.4f}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n5️⃣ 测试生成功能...")
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids[:1],
                max_length=150,
                temperature=0.8
            )
        
        print(f"✅ 生成测试成功")
        print(f"  生成序列长度: {generated_ids.shape[1]}")
        
    except Exception as e:
        print(f"❌ 生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n6️⃣ 显存占用分析...")
    if torch.cuda.is_available():
        print(f"  已分配显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"  已保留显存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    else:
        print("  (使用CPU，无GPU显存统计)")
    
    print("\n" + "="*60)
    print("✅ 模型架构测试全部通过！")
    print("="*60)
    
    print("\n📝 模型架构验证成功，下一步:")
    print("  1. 准备训练数据")
    print("  2. 配置训练参数: configs/training_config.yaml")
    print("  3. 开始训练: python src/train_multimodal.py")
    
    return True

if __name__ == "__main__":
    success = test_model_creation()
    sys.exit(0 if success else 1)
