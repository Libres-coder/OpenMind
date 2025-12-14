import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from .vision_encoder import VisionEncoder
from .multimodal_fusion import MultimodalFusion
class VisionUnderstanding(nn.Module):
    """视觉理解引擎 - 整合视觉编码和多模态融合"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 embed_dim: int = 768, num_heads: int = 8,
                 vision_layers: int = 6, fusion_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 视觉编码器
        self.vision_encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=vision_layers,
            num_heads=num_heads
        )
        
        # 多模态融合
        self.fusion = MultimodalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=fusion_layers
        )
        
        # 任务头
        self.image_classifier = nn.Linear(embed_dim, 1000)  # ImageNet分类
        self.caption_proj = nn.Linear(embed_dim, embed_dim)  # 图像描述
        self.vqa_head = nn.Linear(embed_dim, embed_dim)  # 视觉问答
        
    def encode_image(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码图像"""
        return self.vision_encoder(image)
    
    def fuse_modalities(self, text_features: torch.Tensor,
                        vision_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """融合多模态特征"""
        return self.fusion(text_features, vision_features)
    
    def forward(self, image: torch.Tensor = None,
                text_features: torch.Tensor = None,
                task: str = 'fusion') -> Dict[str, Any]:
        """
        视觉理解前向传播
        
        Args:
            image: 输入图像 [batch, 3, H, W]
            text_features: 文本特征 [batch, seq, embed_dim]
            task: 任务类型 ('encode', 'fusion', 'classify', 'caption', 'vqa')
        """
        results = {}
        
        # 编码图像
        if image is not None:
            vision_output = self.encode_image(image)
            results['vision_cls'] = vision_output['cls_token']
            results['vision_patches'] = vision_output['patch_tokens']
            
            if task == 'classify':
                results['logits'] = self.image_classifier(vision_output['cls_token'])
                return results
        
        # 多模态融合
        if text_features is not None and image is not None:
            fusion_output = self.fuse_modalities(
                text_features, vision_output['patch_tokens']
            )
            results.update(fusion_output)
            
            if task == 'caption':
                results['caption_features'] = self.caption_proj(fusion_output['fused_features'])
            elif task == 'vqa':
                results['vqa_features'] = self.vqa_head(fusion_output['fused_features'])
        
        return results
    
    def get_vision_features(self, image: torch.Tensor) -> torch.Tensor:
        """快速获取视觉特征向量"""
        vision_output = self.encode_image(image)
        return vision_output['cls_token']
print("✅ 视觉理解引擎创建完成")
