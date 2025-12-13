# 立即开始：第一周开发指南 🚀

> **当前阶段**: Week 1-2 核心模型基础
> 
> **本周目标**: 集成改进的视觉编码器，完善训练循环，验证基础功能

---

## 📋 本周任务清单（Week 1）

### Day 1-2: 集成改进的视觉编码器

#### ✅ 任务1: 修改主模型架构

```bash
# 1. 打开主模型文件
code D:\OpenMind\src\model_architecture.py
```

**需要修改的内容**:

```python
# 在文件顶部添加导入
from improved_vision_encoder import (
    ImprovedVisionEncoder,
    ImprovedProjector
)

# 找到 MultimodalReasoningModel 类的 __init__ 方法
# 替换原来的 VisionEncoder

class MultimodalReasoningModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # === 修改这部分 ===
        # 旧代码:
        # self.vision_encoder = VisionEncoder(...)
        
        # 新代码:
        self.vision_encoder = ImprovedVisionEncoder(
            img_size=config.get('img_size', 384),
            patch_size=config.get('patch_size', 14),
            embed_dim=config.get('vision_embed_dim', 1152),
            depth=config.get('vision_depth', 27),
            num_heads=config.get('vision_heads', 16),
            use_flash_attn=True,  # 启用Flash Attention
            qkv_bias=True
        )
        
        # 替换投影层
        # 旧代码:
        # self.vision_projection = nn.Linear(...)
        
        # 新代码:
        self.vision_projector = ImprovedProjector(
            input_dim=config.get('vision_embed_dim', 1152),
            output_dim=config.get('llm_hidden_size', 4096),
            projector_type='token_pooling',  # 使用token pooling
            pooling_kernel=2  # 减少4倍token数量
        )
        
        # 其余代码保持不变...
```

#### ✅ 任务2: 测试新组件

```bash
# 创建测试脚本
cd D:\OpenMind
python -c "
import torch
from src.improved_vision_encoder import ImprovedVisionEncoder, ImprovedProjector

# 测试编码器
encoder = ImprovedVisionEncoder(
    img_size=384,
    embed_dim=1152,
    depth=27,
    use_flash_attn=True
)

# 测试输入
x = torch.randn(2, 3, 384, 384)
features = encoder(x)
print(f'Vision features shape: {features.shape}')

# 测试投影器
projector = ImprovedProjector(
    input_dim=1152,
    output_dim=4096,
    projector_type='token_pooling'
)

projected = projector(features)
print(f'Projected shape: {projected.shape}')
print('✅ 新组件测试通过！')
"
```

预期输出:
```
Vision features shape: torch.Size([2, 729, 1152])
Projected shape: torch.Size([2, 182, 4096])
✅ 新组件测试通过！
```

### Day 3-4: 完善训练循环

#### ✅ 任务3: 升级训练器

创建新文件 `src/production_trainer.py`:

```python
# src/production_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class ProductionTrainer:
    """工程级训练器 - 稳定性和监控增强"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # 学习率调度
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['total_steps']
        )
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', True) else None
        
        # 监控
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 梯度累积
        self.gradient_accumulation_steps = config.get('gradient_accumulation', 4)
        
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        step = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到GPU
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播（混合精度）
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss'] / self.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs['loss'] / self.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 检查梯度异常
                if self._check_gradients():
                    # 优化器步骤
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    step += 1
                else:
                    logger.warning(f"Skipping step {step} due to abnormal gradients")
                    self.optimizer.zero_grad()
            
            # 记录
            total_loss += loss.item() * self.gradient_accumulation_steps
            current_loss = total_loss / (batch_idx + 1)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Loss异常检测
            if self._detect_loss_spike(loss.item()):
                logger.error(f"Loss spike detected at step {step}! Current: {loss.item():.4f}")
                # 可以选择回滚到上一个checkpoint
            
            # 定期保存
            if step % self.config.get('save_steps', 1000) == 0:
                self.save_checkpoint(epoch, step, current_loss)
        
        return total_loss / len(dataloader)
    
    def _check_gradients(self):
        """检查梯度是否正常"""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # 检查NaN或过大的梯度
        if torch.isnan(torch.tensor(total_norm)) or total_norm > 1000:
            return False
        return True
    
    def _detect_loss_spike(self, current_loss):
        """检测Loss突然飙升"""
        self.loss_history.append(current_loss)
        if len(self.loss_history) < 10:
            return False
        
        recent_avg = sum(self.loss_history[-10:]) / 10
        if current_loss > recent_avg * 2.0:
            return True
        return False
    
    def save_checkpoint(self, epoch, step, loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        path = f"{self.config['output_dir']}/checkpoint-epoch{epoch}-step{step}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
        
        # 保存最佳模型
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = f"{self.config['output_dir']}/best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model updated: loss={loss:.4f}")
```

#### ✅ 任务4: 创建训练脚本

创建 `scripts/train_week1.py`:

```python
# scripts/train_week1.py
import torch
import yaml
from pathlib import Path
import sys

# 添加src到路径
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_architecture import MultimodalReasoningModel
from production_trainer import ProductionTrainer
from data_pipeline import MultimodalDataset, collate_fn

def main():
    # 1. 加载配置
    with open('configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. 创建模型
    print("Creating model...")
    model_config = {
        'img_size': 384,
        'vision_embed_dim': 1152,
        'vision_depth': 27,
        'vision_heads': 16,
        'llm_hidden_size': 4096,
        'llm_model_name': config['model']['language_model']
    }
    
    model = MultimodalReasoningModel(model_config)
    model = model.cuda()
    
    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 加载数据
    print("Loading data...")
    train_dataset = MultimodalDataset(
        data_path=config['data']['pretrain_data'],
        image_processor=model.vision_encoder.image_processor,
        tokenizer=model.tokenizer,
        max_length=config['training']['max_length']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    print(f"Dataset loaded. Size: {len(train_dataset)}")
    
    # 4. 创建训练器
    trainer_config = {
        'learning_rate': config['optimizer']['learning_rate'],
        'weight_decay': config['optimizer']['weight_decay'],
        'warmup_steps': config['training']['warmup_steps'],
        'total_steps': len(train_loader) * config['training']['num_epochs'],
        'gradient_accumulation': config['training']['gradient_accumulation'],
        'use_amp': config['training']['mixed_precision'],
        'output_dir': config['training']['output_dir'],
        'save_steps': 500
    }
    
    trainer = ProductionTrainer(model, trainer_config)
    
    # 5. 开始训练
    print("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        avg_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()
```

### Day 5-6: 数据准备和验证

#### ✅ 任务5: 下载基础数据

```bash
# 创建数据下载脚本
# scripts/download_base_data.py
```

```python
# scripts/download_base_data.py
from datasets import load_dataset
import json
from pathlib import Path

def download_coco_subset():
    """下载COCO子集用于初步验证"""
    print("Downloading COCO captions...")
    dataset = load_dataset("HuggingFaceM4/COCO", split="train[:1000]")
    
    # 转换为JSONL格式
    output_path = Path("data/pretrain_small.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for item in dataset:
            sample = {
                'image': item['image_path'],
                'text': item['caption']
            }
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(dataset)} samples to {output_path}")

def download_vqa_subset():
    """下载VQA子集"""
    print("Downloading VQA...")
    dataset = load_dataset("HuggingFaceM4/VQAv2", split="train[:500]")
    
    output_path = Path("data/sft_small.jsonl")
    with open(output_path, 'w') as f:
        for item in dataset:
            sample = {
                'image': item['image_path'],
                'question': item['question'],
                'answer': item['answer']
            }
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(dataset)} samples to {output_path}")

if __name__ == '__main__':
    download_coco_subset()
    download_vqa_subset()
```

运行:
```bash
python scripts/download_base_data.py
```

#### ✅ 任务6: 第一次训练测试

```bash
# 小规模测试训练（确保一切正常）
python scripts/train_week1.py
```

预期输出:
```
Creating model...
Model created. Parameters: 8,234,567,890
Loading data...
Dataset loaded. Size: 1000
Starting training...
Epoch 0: 100%|████████| 250/250 [05:23<00:00, loss=2.3456, lr=1.2e-5]
Epoch 0 completed. Average loss: 2.3456
✅ 训练成功！
```

### Day 7: 基础评估

#### ✅ 任务7: 运行baseline评估

```bash
# 创建简单评估脚本
# scripts/eval_week1.py
```

```python
# scripts/eval_week1.py
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model_architecture import MultimodalReasoningModel
from PIL import Image

def simple_inference_test():
    """简单的推理测试"""
    # 加载模型
    model = MultimodalReasoningModel.from_pretrained('outputs/best_model.pt')
    model = model.cuda()
    model.eval()
    
    # 测试图像理解
    test_image = Image.open('data/test_images/sample.jpg')
    question = "这张图片里有什么？"
    
    with torch.no_grad():
        response = model.generate(
            images=[test_image],
            prompt=question,
            max_length=100
        )
    
    print(f"Question: {question}")
    print(f"Answer: {response}")
    
    return response

if __name__ == '__main__':
    simple_inference_test()
```

---

## 📅 Week 1 时间表

| 日期 | 任务 | 预计时间 | 检查点 |
|------|------|---------|--------|
| Day 1 | 集成改进视觉编码器 | 4小时 | ✅ 组件测试通过 |
| Day 2 | 修改主模型架构 | 4小时 | ✅ 模型加载成功 |
| Day 3 | 创建ProductionTrainer | 6小时 | ✅ 训练器测试通过 |
| Day 4 | 完善训练脚本 | 4小时 | ✅ 脚本可运行 |
| Day 5 | 下载和准备数据 | 3小时 | ✅ 数据准备完成 |
| Day 6 | 第一次训练 | 6小时 | ✅ 训练稳定运行 |
| Day 7 | 基础评估 | 3小时 | ✅ 推理测试通过 |

---

## ✅ Week 1 成功标准

完成以下所有项目即为成功:

- [ ] 改进的视觉编码器集成完成
- [ ] ProductionTrainer创建并测试通过
- [ ] 在1000样本上完成1个epoch训练
- [ ] Loss正常下降（不出现NaN或爆炸）
- [ ] 可以正常保存和加载checkpoint
- [ ] 简单推理测试可以生成文本
- [ ] 显存占用 < 40GB (单卡A100可运行)

---

## 🚨 常见问题和解决方案

### 问题1: CUDA Out of Memory

```python
# 解决方案
# 1. 减小batch size
config['training']['batch_size'] = 1

# 2. 启用梯度累积
config['training']['gradient_accumulation'] = 8

# 3. 启用梯度检查点
model.gradient_checkpointing_enable()

# 4. 使用Flash Attention（已默认开启）
```

### 问题2: Loss不下降

```python
# 检查列表
# 1. 学习率是否太小？
config['optimizer']['learning_rate'] = 2e-5  # 调大

# 2. 数据是否正确？
# 运行: python scripts/test_data_pipeline.py

# 3. 模型是否正确初始化？
# 检查: 打印前几个batch的loss
```

### 问题3: 训练速度慢

```python
# 优化方案
# 1. 启用混合精度
config['training']['mixed_precision'] = True

# 2. 增加num_workers
dataloader = DataLoader(..., num_workers=8)

# 3. 使用更大的batch size + 梯度累积
```

---

## 📊 Week 1 预期结果

完成Week 1后，你应该有:

1. **代码层面**:
   - ✅ 集成了改进的视觉编码器
   - ✅ 稳定的训练循环
   - ✅ 完整的checkpoint管理

2. **模型层面**:
   - ✅ 可以加载和运行的模型
   - ✅ 在小数据集上Loss正常下降
   - ✅ 可以生成基础的图文响应

3. **基础设施**:
   - ✅ 数据加载pipeline
   - ✅ 训练监控和日志
   - ✅ 模型保存和加载

---

## 🎯 下周预告 (Week 2)

Week 2 将focus on:
- 扩大训练数据量（10万+ 样本）
- 完整的benchmark评估
- 性能优化和调优
- 准备进入Week 3的长文本能力开发

---

## 💡 立即开始

```bash
# 现在就执行这些命令！

# 1. 测试改进的组件
cd D:\OpenMind
python src\improved_vision_encoder.py

# 2. 如果测试通过，开始集成
code src\model_architecture.py
# 按照上面的指引修改代码

# 3. 创建训练器
code src\production_trainer.py
# 复制上面的代码

# 4. 准备数据
python scripts\download_base_data.py

# 5. 开始训练！
python scripts\train_week1.py
```

**第一周最重要的是：让训练跑起来并稳定运行！** 🚀
