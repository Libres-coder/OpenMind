"""
OpenMind Agent 端到端训练脚本
用于训练整合了记忆、推理、视觉、进化的统一智能体
"""

import os
import sys
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from pathlib import Path
import random
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import OpenMindAgent, AgentConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型
    hidden_size: int = 768
    max_cot_steps: int = 5
    img_size: int = 224
    patch_size: int = 16
    vision_layers: int = 6
    fusion_layers: int = 4
    
    # 训练
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    
    # 损失权重
    task_loss_weight: float = 1.0
    reasoning_loss_weight: float = 0.1
    evolution_loss_weight: float = 0.05
    contrastive_loss_weight: float = 0.1
    
    # 硬件
    device: str = "auto"
    fp16: bool = True
    seed: int = 42
    
    # 路径
    save_dir: str = "outputs/openmind"
    log_dir: str = "logs/openmind"


class MultimodalDataset(Dataset):
    """多模态数据集"""
    
    def __init__(self, data_file: str, config: TrainingConfig, mode: str = "train"):
        self.config = config
        self.mode = mode
        self.samples = []
        
        # 加载数据
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
            logger.info(f"加载了 {len(self.samples)} 条{mode}数据")
        else:
            # 生成模拟数据用于测试
            logger.warning(f"数据文件不存在: {data_file}，生成模拟数据")
            self.samples = self._generate_mock_data(100 if mode == "train" else 20)
    
    def _generate_mock_data(self, num_samples: int) -> List[Dict]:
        """生成模拟数据"""
        samples = []
        modalities = ["text", "image", "multimodal"]
        
        for i in range(num_samples):
            modality = random.choice(modalities)
            sample = {
                "id": f"sample_{i}",
                "modality": modality,
                "text": f"这是第{i}个样本的文本内容",
                "label": random.randint(0, 9)
            }
            
            if modality in ["image", "multimodal"]:
                sample["has_image"] = True
            
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 生成文本嵌入（模拟，实际应使用tokenizer）
        text_embedding = torch.randn(self.config.hidden_size)
        
        # 生成图像（模拟）
        if sample.get("has_image"):
            image = torch.randn(3, self.config.img_size, self.config.img_size)
        else:
            image = None
        
        # 标签
        label = torch.tensor(sample.get("label", 0))
        
        return {
            "text_embedding": text_embedding,
            "image": image,
            "label": label,
            "modality": sample.get("modality", "text")
        }


def collate_fn(batch):
    """自定义collate函数，处理可选的图像"""
    text_embeddings = torch.stack([item["text_embedding"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    modalities = [item["modality"] for item in batch]
    
    # 处理图像
    images = [item["image"] for item in batch]
    if all(img is not None for img in images):
        images = torch.stack(images)
    else:
        images = None
    
    return {
        "text_embedding": text_embeddings,
        "image": images,
        "label": labels,
        "modality": modalities
    }


class OpenMindTrainer:
    """OpenMind Agent 训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 创建模型
        logger.info("创建OpenMindAgent...")
        agent_config = AgentConfig(
            hidden_size=config.hidden_size,
            max_cot_steps=config.max_cot_steps,
            img_size=config.img_size,
            patch_size=config.patch_size,
            vision_layers=config.vision_layers,
            fusion_layers=config.fusion_layers
        )
        self.model = OpenMindAgent(agent_config).to(self.device)
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数: {total_params/1e6:.2f}M (可训练: {trainable_params/1e6:.2f}M)")
        
        # 创建优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 创建学习率调度器
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps - config.warmup_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[config.warmup_steps]
        )
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 and self.device.type == "cuda" else None
        
        # 创建输出目录
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # 损失函数
        self.task_criterion = nn.CrossEntropyLoss()
        
        # 输出投影层（用于分类任务）
        self.output_proj = nn.Linear(config.hidden_size, 10).to(self.device)
        
    def _setup_device(self) -> torch.device:
        """设置设备"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        logger.info(f"使用设备: {device}")
        return device
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """计算损失"""
        text_embedding = batch["text_embedding"].to(self.device)
        labels = batch["label"].to(self.device)
        image = batch["image"].to(self.device) if batch["image"] is not None else None
        
        # 前向传播（简化模式，禁用耗时的推理和进化模块）
        outputs = self.model(
            text_embedding=text_embedding,
            image=image,
            use_reasoning=False,
            use_evolution=False
        )
        
        # 任务损失（使用输出的投影）
        logits = self.output_proj(outputs["output"])
        task_loss = self.task_criterion(logits, labels)
        
        # 推理一致性损失
        reasoning_loss = torch.tensor(0.0, device=self.device)
        if "reasoning" in outputs:
            cot_output = outputs["reasoning"]["chain_of_thought"]["final_state"]
            reasoning_loss = F.mse_loss(cot_output, outputs["output"])
        
        # 进化损失
        evolution_loss = torch.tensor(0.0, device=self.device)
        if "evolution" in outputs:
            eval_score = outputs["evolution"]["evaluation"]["overall_score"]
            evolution_loss = (1 - eval_score.mean())  # 鼓励高质量输出
        
        # 总损失
        total_loss = (
            self.config.task_loss_weight * task_loss +
            self.config.reasoning_loss_weight * reasoning_loss +
            self.config.evolution_loss_weight * evolution_loss
        )
        
        return {
            "loss": total_loss,
            "task_loss": task_loss,
            "reasoning_loss": reasoning_loss,
            "evolution_loss": evolution_loss
        }
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """单步训练"""
        self.model.train()
        
        # 混合精度训练
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                losses = self.compute_loss(batch)
                loss = losses["loss"] / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            losses = self.compute_loss(batch)
            loss = losses["loss"] / self.config.gradient_accumulation_steps
            loss.backward()
        
        return {k: v.item() for k, v in losses.items()}
    
    def optimizer_step(self):
        """优化器更新"""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1
    
    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0
        total_task_loss = 0
        num_batches = 0
        
        for batch in eval_dataloader:
            losses = self.compute_loss(batch)
            total_loss += losses["loss"].item()
            total_task_loss += losses["task_loss"].item()
            num_batches += 1
        
        return {
            "eval_loss": total_loss / num_batches,
            "eval_task_loss": total_task_loss / num_batches
        }
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config.__dict__
        }
        torch.save(checkpoint, path)
        logger.info(f"保存检查点: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"加载检查点: {path}, step={self.global_step}")
    
    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader = None):
        """训练循环"""
        logger.info("开始训练...")
        logger.info(f"  总步数: {self.config.max_steps}")
        logger.info(f"  批次大小: {self.config.batch_size}")
        logger.info(f"  梯度累积: {self.config.gradient_accumulation_steps}")
        logger.info(f"  学习率: {self.config.learning_rate}")
        
        accumulation_loss = {}
        
        while self.global_step < self.config.max_steps:
            for batch in train_dataloader:
                # 训练步
                losses = self.train_step(batch)
                
                # 累积损失用于日志
                for k, v in losses.items():
                    accumulation_loss[k] = accumulation_loss.get(k, 0) + v
                
                # 梯度累积
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    
                    # 日志
                    if self.global_step % self.config.logging_steps == 0:
                        avg_losses = {k: v / self.config.gradient_accumulation_steps 
                                     for k, v in accumulation_loss.items()}
                        lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step}: "
                            f"loss={avg_losses['loss']:.4f}, "
                            f"task={avg_losses['task_loss']:.4f}, "
                            f"lr={lr:.2e}"
                        )
                        accumulation_loss = {}
                    
                    # 评估
                    if eval_dataloader and self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate(eval_dataloader)
                        logger.info(f"Eval Step {self.global_step}: {eval_metrics}")
                        
                        if eval_metrics["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval_loss"]
                            self.save_checkpoint(
                                os.path.join(self.config.save_dir, "best_model.pt")
                            )
                    
                    # 保存
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(
                            os.path.join(self.config.save_dir, f"checkpoint_{self.global_step}.pt")
                        )
                    
                    if self.global_step >= self.config.max_steps:
                        break
        
        # 保存最终模型
        self.save_checkpoint(os.path.join(self.config.save_dir, "final_model.pt"))
        logger.info("训练完成!")


def load_config(config_path: str) -> TrainingConfig:
    """从YAML加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    config = TrainingConfig()
    
    # 模型配置
    if "model" in yaml_config:
        for k, v in yaml_config["model"].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # 训练配置
    if "training" in yaml_config:
        for k, v in yaml_config["training"].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # 损失配置
    if "loss" in yaml_config:
        for k, v in yaml_config["loss"].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    # 检查点配置
    if "checkpoint" in yaml_config:
        if "save_dir" in yaml_config["checkpoint"]:
            config.save_dir = yaml_config["checkpoint"]["save_dir"]
    
    # 日志配置
    if "logging" in yaml_config:
        if "log_dir" in yaml_config["logging"]:
            config.log_dir = yaml_config["logging"]["log_dir"]
    
    # 硬件配置
    if "hardware" in yaml_config:
        if "device" in yaml_config["hardware"]:
            config.device = yaml_config["hardware"]["device"]
    
    if "seed" in yaml_config:
        config.seed = yaml_config["seed"]
    
    return config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="训练OpenMind Agent")
    parser.add_argument("--config", type=str, default="configs/openmind_train.yaml",
                       help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                       help="恢复训练的检查点路径")
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"从 {args.config} 加载配置")
    else:
        config = TrainingConfig()
        logger.info("使用默认配置")
    
    # 创建数据集
    train_dataset = MultimodalDataset("data/train.jsonl", config, mode="train")
    eval_dataset = MultimodalDataset("data/eval.jsonl", config, mode="eval")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建训练器
    trainer = OpenMindTrainer(config)
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()
