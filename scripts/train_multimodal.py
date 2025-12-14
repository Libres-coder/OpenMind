import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from torch.cuda.amp import autocast, GradScaler
import wandb
import os
from tqdm import tqdm
from typing import Dict, Optional
import json
from datetime import datetime

from model_architecture import create_model
from data_pipeline import create_dataloader


class MultimodalTrainer:
    def __init__(
        self,
        model_config: Dict,
        training_config: Dict,
        output_dir: str = "./checkpoints"
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.accelerator = Accelerator(
            mixed_precision=training_config.get('mixed_precision', 'bf16'),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4)
        )
        
        self.model = create_model(model_config)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.get('learning_rate', 1e-5),
            betas=(0.9, 0.95),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
        
        self.scaler = GradScaler() if training_config.get('use_amp', False) else None
        
        if training_config.get('use_wandb', False):
            wandb.init(
                project=training_config.get('project_name', 'multimodal-training'),
                config={**model_config, **training_config}
            )
    
    def setup_dataloader(self, stage: str = 'pretrain'):
        data_path = self.training_config.get(f'{stage}_data_path')
        
        dataloader = create_dataloader(
            data_path=data_path,
            tokenizer_name=self.model_config['base_model'],
            batch_size=self.training_config.get('batch_size', 8),
            num_workers=self.training_config.get('num_workers', 4),
            shuffle=True,
            enable_audio=self.model_config.get('enable_audio', False)
        )
        
        return dataloader
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        use_reasoning: bool = False
    ) -> float:
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                if self.scaler:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            images=batch.get('images'),
                            audio=batch.get('audio'),
                            labels=batch['labels'],
                            use_reasoning=use_reasoning
                        )
                        loss = outputs['loss']
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        images=batch.get('images'),
                        audio=batch.get('audio'),
                        labels=batch['labels'],
                        use_reasoning=use_reasoning
                    )
                    loss = outputs['loss']
                
                self.accelerator.backward(loss)
                
                if self.training_config.get('max_grad_norm'):
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['max_grad_norm']
                    )
                
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({'loss': loss.item()})
                    
                    if self.training_config.get('use_wandb') and step % 10 == 0:
                        wandb.log({
                            'train/loss': loss.item(),
                            'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                            'train/epoch': epoch
                        })
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    images=batch.get('images'),
                    audio=batch.get('audio'),
                    labels=batch['labels']
                )
                loss = outputs['loss']
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return {'eval_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        if not self.accelerator.is_local_main_process:
            return
        
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        with open(os.path.join(checkpoint_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self, num_epochs: int, stage: str = 'pretrain'):
        dataloader = self.setup_dataloader(stage)
        
        total_steps = len(dataloader) * num_epochs
        warmup_steps = int(total_steps * self.training_config.get('warmup_ratio', 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.model, self.optimizer, self.scheduler, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, dataloader
        )
        
        print(f"\n{'='*50}")
        print(f"Starting {stage} training")
        print(f"Total epochs: {num_epochs}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"{'='*50}\n")
        
        use_reasoning = stage in ['sft', 'rl']
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(dataloader, epoch, use_reasoning)
            
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.accelerator.is_local_main_process:
                print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
                
                if self.training_config.get('use_wandb'):
                    wandb.log({'epoch/train_loss': train_loss})
            
            if (epoch + 1) % self.training_config.get('save_every', 5) == 0:
                self.save_checkpoint(epoch, metrics)
        
        if self.accelerator.is_local_main_process:
            print("\nTraining completed!")
            if self.training_config.get('use_wandb'):
                wandb.finish()


def main():
    model_config = {
        'base_model': 'Qwen/Qwen2-7B',
        'vision_model': 'openai/clip-vit-large-patch14',
        'freeze_vision': True,
        'perceiver_depth': 6,
        'num_latents': 64,
        'enable_audio': False,
        'enable_cot': True,
        'enable_verification': True
    }
    
    training_config = {
        'pretrain_data_path': 'data/pretrain.jsonl',
        'sft_data_path': 'data/sft.jsonl',
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 1e-5,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'warmup_ratio': 0.1,
        'num_workers': 4,
        'mixed_precision': 'bf16',
        'use_amp': False,
        'use_wandb': False,
        'project_name': 'multimodal-reasoning-model',
        'save_every': 5
    }
    
    trainer = MultimodalTrainer(
        model_config=model_config,
        training_config=training_config,
        output_dir="./checkpoints"
    )
    
    print("Stage 1: Multimodal Pretraining")
    trainer.train(num_epochs=10, stage='pretrain')
    
    print("\nStage 2: Supervised Fine-tuning")
    trainer.train(num_epochs=5, stage='sft')


if __name__ == "__main__":
    main()
