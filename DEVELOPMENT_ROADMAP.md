# 多模态模型开发路线图 🗺️

> **当前状态**: 已完成技术方案设计和代码框架
> 
> **下一步**: 开始实施开发

---

## 📅 完整开发时间线（16周）

```
第1周   ✅ 环境搭建 → 基础验证 → 单元测试
第2-3周 📦 数据下载 → 数据处理 → Pipeline验证
第4-5周 🧪 模型测试 → 小规模训练 → Debug优化
第6-8周 🚀 预训练启动 → 持续监控 → 检查点管理
第9-12周 🎯 指令微调 → 能力提升 → 性能调优
第13-16周 📊 全面评估 → 模型优化 → 部署上线
```

---

## 🎯 第1周：环境搭建与验证（立即开始）

### Day 1-2: 环境配置

#### 1.1 安装依赖（Windows）

```powershell
# 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 升级pip
python -m pip install --upgrade pip

# 安装PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 安装核心依赖
pip install transformers==4.36.0 accelerate==0.25.0 datasets==2.16.0

# 安装训练工具（可选）
pip install deepspeed
pip install flash-attn --no-build-isolation  # 需要CUDA和编译环境

# 安装数据处理
pip install webdataset pillow requests tqdm pyyaml

# 安装评估工具
pip install scikit-learn nltk rouge-score wandb tensorboard
```

#### 1.2 验证安装

创建验证脚本：

```python
# scripts/verify_environment.py
import sys
import torch
import transformers
import accelerate

print("="*50)
print("环境验证")
print("="*50)

print(f"\nPython版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"Accelerate版本: {accelerate.__version__}")

print(f"\nCUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

print("\n✅ 环境验证完成！")
```

运行验证：
```bash
python scripts/verify_environment.py
```

### Day 3-4: 测试模型加载

#### 1.3 下载基础模型

```python
# scripts/download_base_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-7B"  # 或使用更小的模型测试
print(f"下载模型: {model_name}")

# 下载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
tokenizer.save_pretrained("./models/qwen2-7b")

# 下载模型（可选：先只下载配置）
from transformers import AutoConfig
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.save_pretrained("./models/qwen2-7b")

print("✅ 模型下载完成")
```

#### 1.4 测试模型架构

```python
# scripts/test_model_architecture.py
import torch
from src.model_architecture import create_model

print("测试模型架构...")

config = {
    'base_model': './models/qwen2-7b',  # 使用本地路径
    'vision_model': 'openai/clip-vit-large-patch14',
    'freeze_vision': True,
    'perceiver_depth': 2,  # 减小测试
    'num_latents': 32,     # 减小测试
    'enable_audio': False,
    'enable_cot': True,
    'enable_verification': True
}

try:
    model = create_model(config)
    print("✅ 模型创建成功")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total_params/1e9:.2f}B")
    print(f"可训练参数: {trainable_params/1e9:.2f}B")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    print("\n测试前向传播...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    print(f"✅ 前向传播成功，输出shape: {outputs['logits'].shape}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
```

### Day 5-7: 数据准备基础

#### 1.5 创建示例数据集

```python
# scripts/create_sample_data.py
import json
import os
from pathlib import Path

def create_sample_pretrain_data():
    """创建预训练示例数据"""
    output_dir = Path("data/sample/pretrain")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    for i in range(100):  # 创建100个示例
        sample = {
            "text": f"这是第{i}个训练样本。包含多模态内容的描述文本。",
            "metadata": {
                "source": "sample",
                "id": i
            }
        }
        samples.append(sample)
    
    # 保存为JSONL
    with open(output_dir / "train.jsonl", 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建了 {len(samples)} 个预训练样本")

def create_sample_sft_data():
    """创建指令微调示例数据"""
    output_dir = Path("data/sample/sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = [
        {
            "conversations": [
                {"from": "user", "value": "你好，请介绍一下你自己。"},
                {"from": "assistant", "value": "你好！我是一个多模态AI助手，可以处理文本、图像和音频等多种输入。"}
            ]
        },
        {
            "conversations": [
                {"from": "user", "value": "请解释什么是机器学习？"},
                {"from": "assistant", "value": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进，而无需明确编程。"}
            ]
        }
    ] * 50  # 复制50次作为示例
    
    with open(output_dir / "train.jsonl", 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 创建了 {len(samples)} 个SFT样本")

if __name__ == "__main__":
    create_sample_pretrain_data()
    create_sample_sft_data()
    print("\n✅ 示例数据创建完成")
```

#### 1.6 测试数据加载

```python
# scripts/test_data_pipeline.py
from src.data_pipeline import create_dataloader

print("测试数据加载...")

try:
    dataloader = create_dataloader(
        data_path="data/sample/pretrain/train.jsonl",
        tokenizer_name="Qwen/Qwen2-7B",
        batch_size=2,
        num_workers=0,  # Windows上设为0
        shuffle=False
    )
    
    # 测试加载一个batch
    for batch in dataloader:
        print(f"✅ 数据加载成功")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        break
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
```

---

## 📦 第2-3周：数据准备（核心任务）

### 2.1 真实数据下载策略

#### 选项A: 小规模快速验证（推荐先做）

```python
# scripts/download_small_datasets.py
"""
下载小规模数据集用于快速验证
- COCO 2017 验证集: ~5GB
- CC3M 子集: ~10GB
"""
import os
from datasets import load_dataset

def download_coco_val():
    """下载COCO验证集"""
    print("下载COCO验证集...")
    dataset = load_dataset("HuggingFaceM4/COCO", split="validation")
    dataset.save_to_disk("data/coco_val")
    print(f"✅ 下载完成: {len(dataset)} 样本")

def download_cc3m_subset():
    """下载CC3M子集"""
    print("下载CC3M子集...")
    dataset = load_dataset("conceptual_captions", split="train[:10000]")
    dataset.save_to_disk("data/cc3m_subset")
    print(f"✅ 下载完成: {len(dataset)} 样本")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    download_coco_val()
    download_cc3m_subset()
```

#### 选项B: 大规模数据准备（正式训练）

参考文档中的数据集清单，使用 `img2dataset` 下载LAION等大规模数据。

### 2.2 数据预处理Pipeline

```python
# scripts/preprocess_data.py
"""
统一数据预处理脚本
"""
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk

def preprocess_coco_for_training(input_dir, output_file):
    """将COCO格式转换为训练格式"""
    dataset = load_from_disk(input_dir)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc="处理COCO数据"):
            # 转换为统一格式
            sample = {
                "text": item['caption'],
                "image": item['image_path'],  # 需要保存图像路径
                "metadata": {
                    "source": "coco",
                    "image_id": item.get('image_id')
                }
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"✅ 预处理完成: {output_file}")

if __name__ == "__main__":
    preprocess_coco_for_training(
        "data/coco_val",
        "data/processed/pretrain_coco.jsonl"
    )
```

---

## 🧪 第4-5周：模型训练验证

### 3.1 超小规模训练测试（2-4小时）

```bash
# 目的：验证整个训练流程没有bug
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage pretrain \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_steps 50 \
    --output_dir checkpoints/test_run
```

预期结果：
- ✅ 训练正常运行
- ✅ Loss下降
- ✅ 检查点正常保存
- ✅ 显存占用正常

### 3.2 小规模完整训练（1-2天）

```bash
# 使用示例数据集完整训练1个epoch
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage pretrain \
    --num_epochs 3 \
    --batch_size 2 \
    --output_dir checkpoints/small_scale
```

### 3.3 训练监控和调试

创建监控脚本：

```python
# scripts/monitor_training.py
"""
实时监控训练进度
"""
import json
from pathlib import Path
import time

def monitor_checkpoints(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    
    while True:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
        
        if checkpoints:
            latest = checkpoints[-1]
            metrics_file = latest / "metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                print(f"\n最新检查点: {latest.name}")
                print(f"Epoch: {metrics.get('epoch')}")
                print(f"Train Loss: {metrics.get('train_loss', 'N/A')}")
                print(f"时间: {metrics.get('timestamp')}")
        
        time.sleep(60)  # 每分钟检查一次

if __name__ == "__main__":
    monitor_checkpoints("checkpoints/small_scale")
```

---

## 🚀 第6-8周：多模态预训练

### 4.1 配置优化

根据你的硬件配置调整 `configs/training_config.yaml`:

```yaml
# 4x RTX 4090 配置示例
training:
  batch_size: 2                    # 每GPU批次
  gradient_accumulation_steps: 8   # 有效batch_size=64
  learning_rate: 2.0e-5
  mixed_precision: "bf16"
  use_gradient_checkpointing: true
  
  num_epochs:
    pretrain: 10
```

### 4.2 启动预训练

```bash
# 单GPU训练
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage pretrain

# 多GPU训练 (使用Accelerate)
accelerate launch --multi_gpu --num_processes=4 \
    src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage pretrain
```

### 4.3 持续监控

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 或启动Wandb（需要先登录）
wandb login
# 然后在配置中启用: use_wandb: true
```

---

## 🎯 第9-12周：指令微调

### 5.1 准备SFT数据

推荐数据源：
- ShareGPT对话数据
- LLaVA-Instruct视觉指令
- 自建高质量数据（最重要）

### 5.2 启动SFT训练

```bash
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage sft \
    --resume_from_checkpoint checkpoints/pretrain/checkpoint-epoch-10
```

---

## 📊 第13-16周：评估和优化

### 6.1 全面评估

```bash
python src/evaluate_model.py \
    --model_config configs/model_config.yaml \
    --checkpoint checkpoints/sft/best_model.pt \
    --eval_config configs/eval_config.yaml
```

### 6.2 模型量化（可选）

```python
# scripts/quantize_model.py
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/sft/best_model",
    device_map="auto",
    load_in_8bit=True  # INT8量化
)

model.save_pretrained("checkpoints/quantized_int8")
```

---

## 🛠️ 开发最佳实践

### 1. 版本控制

```bash
git init
git add .
git commit -m "Initial commit: 多模态模型训练框架"

# 创建开发分支
git checkout -b feature/data-pipeline
```

### 2. 实验记录

创建实验日志：

```python
# experiments/exp_log.md
## 实验1: 基础架构验证
- 日期: 2025-01-XX
- 配置: Qwen2-7B + CLIP
- 数据: COCO val 5K
- 结果: Loss从8.5降到6.2
- 问题: 显存占用过高
- 解决: 启用gradient_checkpointing
```

### 3. 定期检查点

```python
# 每天备份重要检查点
import shutil
from datetime import datetime

checkpoint_dir = "checkpoints/pretrain/checkpoint-epoch-5"
backup_dir = f"backups/{datetime.now().strftime('%Y%m%d')}"
shutil.copytree(checkpoint_dir, backup_dir)
```

---

## ⚠️ 常见问题预防

### 问题1: CUDA Out of Memory

**解决方案**:
```yaml
# 减小batch_size
batch_size: 1
gradient_accumulation_steps: 16

# 启用梯度检查点
use_gradient_checkpointing: true

# 冻结视觉编码器
freeze_vision: true
```

### 问题2: 训练速度慢

**解决方案**:
- 使用更快的数据加载器 (WebDataset)
- 启用Flash Attention
- 增加 num_workers
- 使用混合精度训练

### 问题3: Loss不下降

**检查清单**:
- [ ] 学习率是否合适 (1e-5到5e-5)
- [ ] 数据是否正确加载
- [ ] 标签是否正确对齐
- [ ] 梯度是否正常 (不要梯度爆炸)

---

## 📝 每周检查清单

### ✅ 每周必做
- [ ] 检查训练loss曲线
- [ ] 查看最新检查点性能
- [ ] 备份重要模型
- [ ] 记录实验日志
- [ ] 更新技术文档

### ✅ 每月必做
- [ ] 全面性能评估
- [ ] 代码重构优化
- [ ] 数据质量分析
- [ ] 资源使用优化
- [ ] 技术分享/汇报

---

## 🎯 阶段性目标

### 短期目标（1个月）
- ✅ 完成环境搭建
- ✅ 验证完整训练流程
- ✅ 完成小规模数据预训练
- ✅ 初步评估模型能力

### 中期目标（3个月）
- ✅ 完成多模态预训练
- ✅ 完成指令微调
- ✅ 达到基准性能指标
- ✅ 优化推理速度

### 长期目标（6个月）
- ✅ 达到或超越同类开源模型
- ✅ 完成模型部署
- ✅ 撰写技术报告
- ✅ 开源模型和代码

---

**下一步行动**: 从第1周Day1开始，运行 `python scripts/verify_environment.py` ✨
