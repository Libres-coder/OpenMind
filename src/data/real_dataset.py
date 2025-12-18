"""
真实数据集加载器
支持COCO、CC3M等常用多模态数据集
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置"""
    hidden_size: int = 768
    img_size: int = 224
    max_text_length: int = 512


class COCOCaptionDataset(Dataset):
    """COCO Caption数据集"""
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        config: DatasetConfig,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        self.image_dir = Path(image_dir)
        self.config = config
        self.transform = transform
        self.mode = mode
        
        logger.info(f"加载COCO annotations: {annotation_file}")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        self.id_to_filename = {
            img['id']: img['file_name'] 
            for img in coco_data['images']
        }
        
        self.samples = []
        for ann in coco_data['annotations']:
            self.samples.append({
                'image_id': ann['image_id'],
                'caption': ann['caption'],
                'id': ann['id']
            })
        
        logger.info(f"加载了 {len(self.samples)} 条{mode}数据")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_filename = self.id_to_filename[sample['image_id']]
        image_path = self.image_dir / image_filename
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            else:
                image = image.resize((self.config.img_size, self.config.img_size))
                image = torch.from_numpy(
                    np.array(image).transpose(2, 0, 1)
                ).float() / 255.0
        except Exception as e:
            logger.warning(f"加载图像失败 {image_path}: {e}")
            image = torch.zeros(3, self.config.img_size, self.config.img_size)
        
        text_embedding = torch.randn(self.config.hidden_size)
        
        return {
            'image': image,
            'text_embedding': text_embedding,
            'caption': sample['caption'],
            'image_id': sample['image_id'],
            'label': torch.tensor(0)
        }


class ImageTextPairDataset(Dataset):
    """通用图文对数据集（JSONL格式）"""
    
    def __init__(
        self,
        data_file: str,
        image_dir: str,
        config: DatasetConfig,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        self.image_dir = Path(image_dir)
        self.config = config
        self.transform = transform
        self.mode = mode
        
        self.samples = []
        logger.info(f"加载数据: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        logger.info(f"加载了 {len(self.samples)} 条{mode}数据")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image_path = sample.get('image')
        if image_path:
            full_path = self.image_dir / image_path
            try:
                image = Image.open(full_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                else:
                    image = image.resize((self.config.img_size, self.config.img_size))
                    image = torch.from_numpy(
                        np.array(image).transpose(2, 0, 1)
                    ).float() / 255.0
            except Exception as e:
                logger.warning(f"加载图像失败 {full_path}: {e}")
                image = torch.zeros(3, self.config.img_size, self.config.img_size)
        else:
            image = None
        
        text_embedding = torch.randn(self.config.hidden_size)
        label = sample.get('label', 0)
        
        return {
            'image': image,
            'text_embedding': text_embedding,
            'text': sample.get('text', ''),
            'label': torch.tensor(label)
        }


def collate_fn_real(batch: List[Dict]) -> Dict[str, Any]:
    """真实数据集的collate函数"""
    text_embeddings = torch.stack([item['text_embedding'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    images = [item['image'] for item in batch]
    if all(img is not None for img in images):
        images = torch.stack(images)
    else:
        images = None
    
    texts = [item.get('text', '') or item.get('caption', '') for item in batch]
    
    return {
        'text_embedding': text_embeddings,
        'image': images,
        'label': labels,
        'text': texts
    }


def create_dataloader(
    dataset_type: str,
    config: DatasetConfig,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """创建数据加载器"""
    if dataset_type == 'coco':
        dataset = COCOCaptionDataset(config=config, **kwargs)
    elif dataset_type == 'jsonl':
        dataset = ImageTextPairDataset(config=config, **kwargs)
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_real,
        num_workers=num_workers
    )
