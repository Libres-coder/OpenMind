"""
Week 1 训练脚本 - 基础训练循环验证
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import logging
from model_architecture import MultimodalReasoningModel
from production_trainer import ProductionTrainer
from PIL import Image
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleMultimodalDataset(Dataset):
    """简单的多模态数据集"""
    
    def __init__(self, data_path: str, image_dir: str, tokenizer, max_length: int = 512):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.samples = []
        if self.data_path.exists():
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        
        logger.info(f"加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 文本处理
        text = sample.get('text', 'This is a test sample.')
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 图像处理
        image = None
        if 'image' in sample and self.image_dir.exists():
            image_path = self.image_dir / sample['image']
            if image_path.exists():
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize((384, 384))
                    image = torch.tensor(list(image.getdata())).reshape(3, 384, 384).float() / 255.0
                except Exception as e:
                    logger.warning(f"图像加载失败 {image_path}: {e}")
        
        # 如果没有图像，创建随机图像
        if image is None:
            image = torch.randn(3, 384, 384)
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'images': image,
            'labels': tokens['input_ids'].squeeze(0)
        }


def collate_fn(batch):
    """批处理函数"""
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'images': torch.stack([x['images'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


def main():
    """主训练函数"""
    # 加载配置
    config_path = Path('configs/week1_training.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("="*60)
    logger.info("Week 1 训练开始")
    logger.info("="*60)
    
    # 创建模型
    logger.info("\n[1/5] 创建模型...")
    model_config = config['model']
    model = MultimodalReasoningModel(model_config)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    logger.info(f"模型参数: {total_params:.2f}B (可训练: {trainable_params:.2f}B)")
    
    # 创建数据集
    logger.info("\n[2/5] 加载数据集...")
    data_config = config['data']
    train_config = config['training']
    
    train_dataset = SimpleMultimodalDataset(
        data_config['train_data'],
        data_config['image_dir'],
        model.tokenizer,
        train_config['max_length']
    )
    
    if len(train_dataset) == 0:
        logger.warning("⚠️ 训练数据为空，将创建10个示例数据进行测试...")
        # 创建示例数据
        Path(data_config['train_data']).parent.mkdir(parents=True, exist_ok=True)
        with open(data_config['train_data'], 'w', encoding='utf-8') as f:
            for i in range(10):
                sample = {
                    'text': f'This is training sample {i}. The image shows a test case.',
                    'image': None
                }
                f.write(json.dumps(sample) + '\n')
        
        # 重新加载
        train_dataset = SimpleMultimodalDataset(
            data_config['train_data'],
            data_config['image_dir'],
            model.tokenizer,
            train_config['max_length']
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=0,  # Windows上使用0
        collate_fn=collate_fn
    )
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"批次数: {len(train_dataloader)}")
    
    # 创建训练器
    logger.info("\n[3/5] 初始化训练器...")
    trainer_config = {
        **train_config,
        'total_steps': train_config['max_steps']
    }
    trainer = ProductionTrainer(model, trainer_config)
    
    # 开始训练
    logger.info("\n[4/5] 开始训练...")
    logger.info(f"训练轮数: {train_config['num_epochs']}")
    logger.info(f"有效批次大小: {train_config['batch_size'] * train_config['gradient_accumulation']}")
    
    try:
        for epoch in range(train_config['num_epochs']):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{train_config['num_epochs']}")
            logger.info(f"{'='*60}")
            
            epoch_loss = trainer.train_epoch(train_dataloader, epoch + 1)
            
            logger.info(f"\nEpoch {epoch + 1} 完成:")
            logger.info(f"  平均Loss: {epoch_loss:.4f}")
            logger.info(f"  最佳Loss: {trainer.best_loss:.4f}")
            logger.info(f"  全局步数: {trainer.global_step}")
        
        logger.info("\n[5/5] 训练完成！")
        logger.info(f"✅ 最终模型保存在: {trainer.output_dir / 'best_model.pt'}")
        logger.info(f"✅ 训练日志: {trainer.output_dir / 'training_log.json'}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ 训练被中断")
        logger.info(f"当前进度已保存")
    except Exception as e:
        logger.error(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
