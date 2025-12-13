"""
创建示例训练数据
用于快速验证训练pipeline
"""
import json
import os
from pathlib import Path

def create_sample_pretrain_data():
    """创建预训练示例数据"""
    output_dir = Path("data/sample/pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("创建预训练示例数据...")
    
    samples = []
    for i in range(100):
        sample = {
            "text": f"这是第{i}个训练样本。这个样本包含了多模态内容的描述文本，用于验证模型的基本训练流程。",
            "metadata": {
                "source": "sample",
                "id": i,
                "quality": 1.0
            }
        }
        samples.append(sample)
    
    output_file = output_dir / "train.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建了 {len(samples)} 个预训练样本")
    print(f"   文件位置: {output_file}")
    return output_file

def create_sample_sft_data():
    """创建指令微调示例数据"""
    output_dir = Path("data/sample/sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n创建指令微调示例数据...")
    
    conversation_templates = [
        {
            "conversations": [
                {"from": "user", "value": "你好，请介绍一下你自己。"},
                {"from": "assistant", "value": "你好！我是一个多模态AI助手，可以处理文本、图像和音频等多种输入，帮助你完成各种任务。"}
            ]
        },
        {
            "conversations": [
                {"from": "user", "value": "请解释什么是机器学习？"},
                {"from": "assistant", "value": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式和规律，并基于这些学习做出预测或决策，而无需明确编程每一个步骤。"}
            ]
        },
        {
            "conversations": [
                {"from": "user", "value": "如何提高模型的推理能力？"},
                {"from": "assistant", "value": "提高模型推理能力的方法包括：1) 使用思维链(Chain-of-Thought)提示；2) 增加推理任务的训练数据；3) 采用强化学习进行优化；4) 引入自我验证机制。"}
            ]
        },
        {
            "conversations": [
                {"from": "user", "value": "多模态模型有什么优势？"},
                {"from": "assistant", "value": "多模态模型的优势包括：1) 能够同时处理多种类型的输入；2) 可以理解不同模态之间的关联；3) 在复杂任务上表现更好；4) 更接近人类的感知和理解方式。"}
            ]
        },
        {
            "conversations": [
                {"from": "user", "value": "请用简单的话解释什么是神经网络。"},
                {"from": "assistant", "value": "神经网络是一种模仿人脑工作方式的计算模型。它由许多相互连接的\"神经元\"组成，这些神经元可以学习识别数据中的模式，就像人脑学习识别物体和概念一样。"}
            ]
        }
    ]
    
    samples = conversation_templates * 20
    
    output_file = output_dir / "train.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建了 {len(samples)} 个SFT样本")
    print(f"   文件位置: {output_file}")
    return output_file

def create_sample_eval_data():
    """创建评估示例数据"""
    output_dir = Path("data/sample/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n创建评估示例数据...")
    
    samples = [
        {
            "conversations": [
                {"from": "user", "value": "1+1等于几？"},
                {"from": "assistant", "value": "2"}
            ]
        },
        {
            "conversations": [
                {"from": "user", "value": "请列举三种常见的编程语言。"},
                {"from": "assistant", "value": "Python、JavaScript、Java"}
            ]
        }
    ] * 10
    
    output_file = output_dir / "eval.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建了 {len(samples)} 个评估样本")
    print(f"   文件位置: {output_file}")
    return output_file

def main():
    print("="*60)
    print("🎨 创建示例训练数据")
    print("="*60)
    
    create_sample_pretrain_data()
    create_sample_sft_data()
    create_sample_eval_data()
    
    print("\n" + "="*60)
    print("✅ 所有示例数据创建完成！")
    print("="*60)
    
    print("\n📁 数据文件位置:")
    print("  预训练: data/sample/pretrain/train.jsonl")
    print("  指令微调: data/sample/sft/train.jsonl")
    print("  评估: data/sample/eval/eval.jsonl")
    
    print("\n📝 下一步:")
    print("  运行: python scripts/test_data_pipeline.py")

if __name__ == "__main__":
    main()
