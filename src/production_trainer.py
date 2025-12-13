"""
生产级训练器 - 稳定性和监控增强
包含：梯度监控、Loss检测、自动恢复、混合精度训练
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
import json
from typing import Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionTrainer:
    """工程级训练器 - 稳定性和监控增强"""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # 学习率调度
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.get('warmup_steps', 100),
            num_training_steps=config.get('total_steps', 10000)
        )
        
        # 混合精度
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # 监控指标
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        
        # 梯度累积
        self.gradient_accumulation_steps = config.get('gradient_accumulation', 4)
        
        # 输出目录
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("="*60)
        logger.info("ProductionTrainer 初始化完成")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  学习率: {config.get('learning_rate', 2e-5)}")
        logger.info(f"  梯度累积: {self.gradient_accumulation_steps}")
        logger.info(f"  混合精度: {self.use_amp}")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info("="*60)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        accumulation_step = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # 移动数据到GPU
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播（混合精度）
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.get('loss', outputs.get('logits'))
                        if not isinstance(loss, torch.Tensor):
                            # 如果没有loss，计算简单的语言模型loss
                            logits = outputs['logits']
                            labels = batch.get('labels', batch.get('input_ids'))
                            loss = nn.functional.cross_entropy(
                                logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                ignore_index=-100
                            )
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs.get('loss', outputs.get('logits'))
                    if not isinstance(loss, torch.Tensor):
                        logits = outputs['logits']
                        labels = batch.get('labels', batch.get('input_ids'))
                        loss = nn.functional.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100
                        )
                    loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_step += 1
                
                # 梯度累积后更新
                if accumulation_step % self.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0
                    )
                    
                    # 检查梯度异常
                    if self._check_gradients(grad_norm):
                        # 优化器步骤
                        if self.use_amp:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        
                        self.scheduler.step()
                        self.global_step += 1
                    else:
                        logger.warning(f"跳过step {self.global_step}: 梯度异常 (norm={grad_norm:.2f})")
                    
                    self.optimizer.zero_grad()
                
                # 记录
                current_loss = loss.item() * self.gradient_accumulation_steps
                total_loss += current_loss
                avg_loss = total_loss / (batch_idx + 1)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'step': self.global_step
                })
                
                # Loss异常检测
                if self._detect_loss_spike(current_loss):
                    logger.warning(f"⚠️ Loss突增检测: step {self.global_step}, loss={current_loss:.4f}")
                
                # 定期保存
                if self.global_step % self.config.get('save_steps', 1000) == 0:
                    self.save_checkpoint(epoch, avg_loss)
                
            except Exception as e:
                logger.error(f"训练步骤 {batch_idx} 出错: {e}")
                continue
        
        return total_loss / len(dataloader)
    
    def _check_gradients(self, grad_norm: float) -> bool:
        """检查梯度是否正常"""
        # 检查NaN或过大的梯度
        if torch.isnan(torch.tensor(grad_norm)) or grad_norm > 100.0:
            return False
        return True
    
    def _detect_loss_spike(self, current_loss: float) -> bool:
        """检测Loss突然飙升"""
        self.loss_history.append(current_loss)
        
        # 保持最近100个loss
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
        
        if len(self.loss_history) < 10:
            return False
        
        recent_avg = np.mean(self.loss_history[-10:])
        
        # 如果当前loss是最近平均值的2倍以上
        if current_loss > recent_avg * 2.0 and recent_avg > 0:
            return True
        
        return False
    
    def save_checkpoint(self, epoch: int, loss: float):
        """保存检查点"""
        checkpoint_dir = self.output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'loss_history': self.loss_history
        }
        
        # 保存当前checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint-step{self.global_step}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint保存: {checkpoint_path}")
        
        # 保存最佳模型
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"🏆 最佳模型更新: loss={loss:.4f}")
            
            # 保存训练日志
            self._save_training_log(epoch, loss)
    
    def _save_training_log(self, epoch: int, loss: float):
        """保存训练日志"""
        log_path = self.output_dir / 'training_log.json'
        
        log_entry = {
            'epoch': epoch,
            'global_step': self.global_step,
            'loss': loss,
            'best_loss': self.best_loss,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        # 追加到日志文件
        logs = []
        if log_path.exists():
            with open(log_path, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['loss']
        self.loss_history = checkpoint.get('loss_history', [])
        
        logger.info(f"✅ 从checkpoint恢复: {checkpoint_path}")
        logger.info(f"   Epoch: {checkpoint['epoch']}, Step: {self.global_step}, Loss: {checkpoint['loss']:.4f}")
        
        return checkpoint['epoch']


class TrainingConfig:
    """训练配置类"""
    
    @staticmethod
    def get_default_config():
        """获取默认配置"""
        return {
            # 优化器
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'warmup_steps': 500,
            
            # 训练
            'num_epochs': 3,
            'batch_size': 2,
            'gradient_accumulation': 8,
            'use_amp': True,
            
            # 保存和日志
            'output_dir': 'outputs',
            'save_steps': 500,
            'logging_steps': 10,
            
            # 数据
            'max_length': 512,
            'num_workers': 4,
        }
    
    @staticmethod
    def from_yaml(yaml_path: str):
        """从YAML文件加载配置"""
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config


if __name__ == '__main__':
    # 测试训练器创建
    print("测试ProductionTrainer...")
    
    config = TrainingConfig.get_default_config()
    config['total_steps'] = 1000
    
    # 创建简单模型测试
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, input_ids, **kwargs):
            x = torch.randn(2, 10)
            logits = self.linear(x)
            return {'logits': logits}
    
    model = DummyModel()
    trainer = ProductionTrainer(model, config)
    
    print("✅ ProductionTrainer创建成功！")
