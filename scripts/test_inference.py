"""
OpenMind Agent 推理测试脚本
用于验证训练后的模型可以正常进行推理
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import OpenMindAgent, AgentConfig


def load_model_from_checkpoint(checkpoint_path: str, device: str = "auto"):
    """从checkpoint加载模型"""
    print(f"加载checkpoint: {checkpoint_path}")
    
    # 设置设备
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"使用设备: {device}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"Global step: {checkpoint['global_step']}")
    print(f"Best eval loss: {checkpoint.get('best_eval_loss', 'N/A')}")
    
    # 从checkpoint获取配置
    config_dict = checkpoint.get('config', {})
    
    # 创建模型配置
    agent_config = AgentConfig(
        hidden_size=config_dict.get('hidden_size', 768),
        max_cot_steps=config_dict.get('max_cot_steps', 5),
        img_size=config_dict.get('img_size', 224),
        patch_size=config_dict.get('patch_size', 16),
        vision_layers=config_dict.get('vision_layers', 6),
        fusion_layers=config_dict.get('fusion_layers', 4)
    )
    
    # 创建模型
    print("\n创建OpenMindAgent...")
    model = OpenMindAgent(agent_config).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载output_proj（如果存在）
    output_proj = None
    if 'output_proj_state_dict' in checkpoint:
        import torch.nn as nn
        output_proj = nn.Linear(config_dict.get('hidden_size', 768), 10).to(device)
        output_proj.load_state_dict(checkpoint['output_proj_state_dict'])
        output_proj.eval()
        print("✅ output_proj加载成功")
    
    print("✅ 模型加载成功")
    
    return model, output_proj, device, config_dict


def test_text_only_inference(model, output_proj, device, hidden_size):
    """测试纯文本推理"""
    print("\n" + "="*50)
    print("测试1: 纯文本推理")
    print("="*50)
    
    # 创建模拟文本嵌入
    batch_size = 2
    text_embedding = torch.randn(batch_size, hidden_size).to(device)
    
    with torch.no_grad():
        outputs = model(
            text_embedding=text_embedding,
            image=None,
            use_reasoning=True,
            use_evolution=True
        )
    
    print(f"输出keys: {list(outputs.keys())}")
    print(f"输出shape: {outputs['output'].shape}")
    
    if output_proj is not None:
        logits = output_proj(outputs['output'])
        predictions = torch.argmax(logits, dim=-1)
        print(f"预测类别: {predictions.tolist()}")
    
    # 检查各模块输出
    if 'memory' in outputs:
        print(f"记忆模块: ✅ 已启用")
    if 'reasoning' in outputs:
        print(f"推理模块: ✅ 已启用")
        if 'chain_of_thought' in outputs['reasoning']:
            cot = outputs['reasoning']['chain_of_thought']
            print(f"  - CoT步数: {cot.get('num_steps', 'N/A')}")
    if 'evolution' in outputs:
        print(f"进化模块: ✅ 已启用")
        if 'evaluation' in outputs['evolution']:
            eval_score = outputs['evolution']['evaluation']['overall_score']
            print(f"  - 评估分数: {eval_score.mean().item():.4f}")
    
    print("✅ 纯文本推理测试通过")
    return True


def test_multimodal_inference(model, output_proj, device, config_dict):
    """测试多模态推理"""
    print("\n" + "="*50)
    print("测试2: 多模态推理（文本+图像）")
    print("="*50)
    
    hidden_size = config_dict.get('hidden_size', 768)
    img_size = config_dict.get('img_size', 224)
    
    # 创建模拟输入
    batch_size = 2
    text_embedding = torch.randn(batch_size, hidden_size).to(device)
    image = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    with torch.no_grad():
        outputs = model(
            text_embedding=text_embedding,
            image=image,
            use_reasoning=True,
            use_evolution=True
        )
    
    print(f"输出keys: {list(outputs.keys())}")
    print(f"输出shape: {outputs['output'].shape}")
    
    if output_proj is not None:
        logits = output_proj(outputs['output'])
        predictions = torch.argmax(logits, dim=-1)
        print(f"预测类别: {predictions.tolist()}")
    
    # 检查视觉模块
    if 'vision' in outputs:
        print(f"视觉模块: ✅ 已启用")
        vision_out = outputs['vision']
        if 'visual_features' in vision_out:
            print(f"  - 视觉特征shape: {vision_out['visual_features'].shape}")
    
    print("✅ 多模态推理测试通过")
    return True


def test_reasoning_steps(model, device, hidden_size):
    """测试推理步骤"""
    print("\n" + "="*50)
    print("测试3: 推理链详情")
    print("="*50)
    
    text_embedding = torch.randn(1, hidden_size).to(device)
    
    with torch.no_grad():
        outputs = model(
            text_embedding=text_embedding,
            image=None,
            use_reasoning=True,
            use_evolution=False
        )
    
    if 'reasoning' in outputs:
        reasoning = outputs['reasoning']
        print(f"推理模块输出keys: {list(reasoning.keys())}")
        
        if 'chain_of_thought' in reasoning:
            cot = reasoning['chain_of_thought']
            print(f"CoT keys: {list(cot.keys())}")
            if 'final_state' in cot:
                print(f"最终状态shape: {cot['final_state'].shape}")
        
        if 'verification' in reasoning:
            print(f"验证结果: {reasoning['verification']}")
    
    print("✅ 推理链测试通过")
    return True


def test_memory_system(model, device, hidden_size):
    """测试记忆系统"""
    print("\n" + "="*50)
    print("测试4: 记忆系统")
    print("="*50)
    
    text_embedding = torch.randn(1, hidden_size).to(device)
    
    # 第一次推理
    with torch.no_grad():
        outputs1 = model(
            text_embedding=text_embedding,
            image=None,
            use_reasoning=False,
            use_evolution=False
        )
    
    if 'memory' in outputs1:
        memory = outputs1['memory']
        print(f"记忆模块输出keys: {list(memory.keys())}")
        
        if 'short_term' in memory:
            print(f"短期记忆: ✅")
        if 'long_term' in memory:
            print(f"长期记忆: ✅")
    
    print("✅ 记忆系统测试通过")
    return True


def run_all_tests(checkpoint_path: str, device: str = "auto"):
    """运行所有测试"""
    print("="*60)
    print("OpenMind Agent 推理测试")
    print("="*60)
    
    # 加载模型
    model, output_proj, device, config_dict = load_model_from_checkpoint(
        checkpoint_path, device
    )
    
    hidden_size = config_dict.get('hidden_size', 768)
    
    # 运行测试
    results = {}
    
    try:
        results['text_only'] = test_text_only_inference(
            model, output_proj, device, hidden_size
        )
    except Exception as e:
        print(f"❌ 纯文本推理测试失败: {e}")
        results['text_only'] = False
    
    try:
        results['multimodal'] = test_multimodal_inference(
            model, output_proj, device, config_dict
        )
    except Exception as e:
        print(f"❌ 多模态推理测试失败: {e}")
        results['multimodal'] = False
    
    try:
        results['reasoning'] = test_reasoning_steps(model, device, hidden_size)
    except Exception as e:
        print(f"❌ 推理链测试失败: {e}")
        results['reasoning'] = False
    
    try:
        results['memory'] = test_memory_system(model, device, hidden_size)
    except Exception as e:
        print(f"❌ 记忆系统测试失败: {e}")
        results['memory'] = False
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！模型可以正常进行推理。")
    else:
        print("\n⚠️ 部分测试失败，请检查模型。")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="测试OpenMind Agent推理")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Checkpoint文件路径")
    parser.add_argument("--device", type=str, default="auto",
                       help="设备 (auto/cuda/cpu)")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint不存在: {args.checkpoint}")
        sys.exit(1)
    
    success = run_all_tests(args.checkpoint, args.device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
