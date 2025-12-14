"""
Week 1 训练 - 仅验证训练流程（不加载预训练LLM）
适用于内存受限环境
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import logging
from improved_vision_encoder import ImprovedVisionEncoder, ImprovedProjector
from production_trainer import ProductionTrainer
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalReasoningModel(nn.Module):
    """轻量级多模态模型 - 用于训练"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 视觉编码器
        self.vision_encoder = ImprovedVisionEncoder(
            img_size=384,
            patch_size=14,
            embed_dim=512,  # 减小到512
            depth=6,        # 减小到6层
            num_heads=8,
            use_flash_attn=False  # CPU环境关闭
        )
        
        # 投影器
        self.vision_projection = ImprovedProjector(
            input_dim=512,
            output_dim=512,
            projector_type='token_pooling'
        )
        
        # 简单的文本embedding（不用预训练）
        self.text_embedding = nn.Embedding(1000, 512)
        
        # 小型Transformer（3层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 输出头
        self.lm_head = nn.Linear(512, 1000)
        
        logger.info(f"轻量级模型创建完成")
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(f"总参数: {total_params:.1f}M")
    
    def forward(self, input_ids, images, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        
        # 视觉特征
        vision_features = self.vision_encoder(images)
        vision_embeds = self.vision_projection(vision_features)
        
        # 文本embedding
        text_embeds = self.text_embedding(input_ids)
        
        # 拼接 [vision, text]
        combined = torch.cat([vision_embeds, text_embeds], dim=1)
        
        # Transformer
        hidden = self.transformer(combined)
        
        # 预测
        logits = self.lm_head(hidden)
        
        # 计算loss
        loss = None
        if labels is not None:
            # 只对文本部分计算loss
            text_logits = logits[:, vision_embeds.shape[1]:, :]
            loss = nn.functional.cross_entropy(
                text_logits.reshape(-1, 1000),
                labels.reshape(-1),
                ignore_index=-100
            )
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden
        }


class SimpleDataset(Dataset):
    """超简单数据集"""
    
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000, (64,)),
            'images': torch.randn(3, 384, 384),
            'labels': torch.randint(0, 1000, (64,))
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'images': torch.stack([x['images'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


def main():
    logger.info("="*60)
    logger.info("Week 1 训练 - ")
    logger.info("="*60)
    
    # 创建轻量级模型
    logger.info("\n[1/4] 创建轻量级模型...")
    model = MultimodalReasoningModel({})
    model = model.to("cuda")
    
    # 创建数据集
    logger.info("\n[2/4] 创建数据集...")
    dataset = SimpleDataset(num_samples=50)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"批次数: {len(dataloader)}")
    
    # 训练配置
    logger.info("\n[3/4] 配置训练器...")
    config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 10,
        'total_steps': 100,
        'batch_size': 2,
        'gradient_accumulation': 4,
        'use_amp': False,  # CPU环境关闭
        'output_dir': 'outputs/week1',
        'save_steps': 25,
        'logging_steps': 5
    }
    
    trainer = ProductionTrainer(model, config)
    
    # 训练
    logger.info("\n[4/4] 开始训练...")
    logger.info(f"训练3个epoch，每个epoch {len(dataloader)} 批次")
    
    try:
        for epoch in range(3):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/3")
            logger.info(f"{'='*60}")
            
            epoch_loss = trainer.train_epoch(dataloader, epoch + 1)
            
            logger.info(f"\n✅ Epoch {epoch + 1} 完成!")
            logger.info(f"  平均Loss: {epoch_loss:.4f}")
            logger.info(f"  最佳Loss: {trainer.best_loss:.4f}")
            logger.info(f"  全局步数: {trainer.global_step}")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 训练成功！")
        logger.info("="*60)
        logger.info(f"✅ 模型保存: {trainer.output_dir / 'best_model.pt'}")
        logger.info(f"✅ 日志文件: {trainer.output_dir / 'training_log.json'}")
        logger.info("\n接下来可以：")
        logger.info("  1. 增加Windows虚拟内存后加载真实LLM")
        logger.info("  2. 转到GPU环境进行完整训练")
        
    except Exception as e:
        logger.error(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
