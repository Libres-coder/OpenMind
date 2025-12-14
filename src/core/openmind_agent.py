import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
# 导入所有子系统
from ..memory import MemoryManager
from ..reasoning import ReasoningEngine
from ..vision import VisionUnderstanding
from ..evolution import EvolutionEngine
@dataclass
class AgentConfig:
    """Agent配置"""
    hidden_size: int = 768
    max_cot_steps: int = 5
    img_size: int = 224
    patch_size: int = 16
    vision_layers: int = 6
    fusion_layers: int = 4
    buffer_capacity: int = 10000
    memory_path: str = "memory_store"
class OpenMindAgent(nn.Module):
    """
    OpenMind Agent - 整合记忆、推理、视觉、进化的统一智能体
    
    能力:
    - 记忆: 短期对话 + 长期知识存储
    - 推理: Chain-of-Thought + 自我验证 + 问题分解
    - 视觉: 图像编码 + 多模态融合
    - 进化: 自我评估 + 经验回放 + 自我改进
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__()
        
        if config is None:
            config = AgentConfig()
        self.config = config
        
        # 1. 记忆系统 (非神经网络)
        self.memory = MemoryManager(
            max_short_term_turns=10,
            storage_path=config.memory_path
        )
        
        # 2. 推理系统
        self.reasoning = ReasoningEngine(
            hidden_size=config.hidden_size,
            max_cot_steps=config.max_cot_steps
        )
        
        # 3. 视觉系统
        self.vision = VisionUnderstanding(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.hidden_size,
            vision_layers=config.vision_layers,
            fusion_layers=config.fusion_layers
        )
        
        # 4. 进化系统
        self.evolution = EvolutionEngine(
            hidden_size=config.hidden_size,
            buffer_capacity=config.buffer_capacity
        )
        
        # 输入投影器
        self.text_projector = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 输出生成器
        self.output_generator = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # 模态融合门控
        self.modality_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
    def process_text(self, text_embedding: torch.Tensor,
                     use_memory: bool = True) -> Dict[str, Any]:
        """处理文本输入"""
        results = {}
        
        # 投影
        text_features = self.text_projector(text_embedding)
        results['text_features'] = text_features
        
        # 推理
        reasoning_result = self.reasoning(
            text_features.unsqueeze(1) if text_features.dim() == 2 else text_features,
            text_features if text_features.dim() == 2 else text_features.mean(dim=1)
        )
        results['reasoning'] = reasoning_result
        
        return results
    
    def process_image(self, image: torch.Tensor,
                      text_features: torch.Tensor = None) -> Dict[str, Any]:
        """处理图像输入"""
        results = {}
        
        # 视觉编码
        if text_features is not None:
            # 多模态融合
            vision_result = self.vision(
                image, 
                text_features.unsqueeze(1) if text_features.dim() == 2 else text_features,
                task='fusion'
            )
        else:
            vision_result = self.vision(image, task='encode')
            
        results['vision'] = vision_result
        return results
    
    def forward(self, 
                text_embedding: torch.Tensor = None,
                image: torch.Tensor = None,
                mode: str = 'auto',
                use_reasoning: bool = True,
                use_evolution: bool = True) -> Dict[str, Any]:
        """
        统一前向传播
        
        Args:
            text_embedding: 文本嵌入 [batch, hidden] 或 [batch, seq, hidden]
            image: 图像输入 [batch, 3, H, W]
            mode: 'text', 'image', 'multimodal', 'auto'
            use_reasoning: 是否使用推理系统
            use_evolution: 是否使用进化系统
        """
        results = {}
        batch_size = text_embedding.shape[0] if text_embedding is not None else image.shape[0]
        device = text_embedding.device if text_embedding is not None else image.device
        
        # 自动检测模式
        if mode == 'auto':
            if text_embedding is not None and image is not None:
                mode = 'multimodal'
            elif image is not None:
                mode = 'image'
            else:
                mode = 'text'
        
        results['mode'] = mode
        
        # 处理文本
        text_features = None
        if text_embedding is not None:
            text_result = self.process_text(text_embedding)
            results.update(text_result)
            text_features = text_result['text_features']
            if text_features.dim() == 3:
                text_features = text_features.mean(dim=1)
        
        # 处理图像
        vision_features = None
        if image is not None:
            vision_result = self.process_image(image, text_features)
            results.update(vision_result)
            if 'fused_features' in vision_result['vision']:
                vision_features = vision_result['vision']['fused_features']
            else:
                vision_features = vision_result['vision']['vision_cls']
        
        # 融合特征
        if text_features is not None and vision_features is not None:
            # 多模态门控融合
            gate = self.modality_gate(torch.cat([text_features, vision_features], dim=-1))
            fused = gate * text_features + (1 - gate) * vision_features
            results['fused_features'] = fused
            main_features = fused
        elif text_features is not None:
            main_features = text_features
        else:
            main_features = vision_features
        
        # 推理增强
        if use_reasoning and 'reasoning' in results:
            reasoning_output = results['reasoning']['final_output']
            enhanced = torch.cat([main_features, reasoning_output, main_features * reasoning_output], dim=-1)
            output = self.output_generator(enhanced)
        else:
            output = main_features
            
        results['output'] = output
        
        # 进化评估
        if use_evolution:
            input_emb = text_features if text_features is not None else vision_features
            evolution_result = self.evolution(
                input_emb, output, mode='evaluate'
            )
            results['evolution'] = evolution_result
        
        return results
    
    def chat(self, user_input: str, text_embedding: torch.Tensor,
             image: torch.Tensor = None) -> Dict[str, Any]:
        """对话接口"""
        # 添加到记忆
        self.memory.add_conversation("user", user_input)
        
        # 处理
        results = self.forward(text_embedding, image)
        
        # 获取上下文
        context = self.memory.get_context_for_inference(user_input)
        results['memory_context'] = context
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Agent统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': f"{total_params/1e6:.2f}M",
            'trainable_parameters': f"{trainable_params/1e6:.2f}M",
            'memory_stats': self.memory.get_stats(),
            'evolution_stats': self.evolution.get_evolution_summary(),
            'components': {
                'reasoning': f"{sum(p.numel() for p in self.reasoning.parameters())/1e6:.2f}M",
                'vision': f"{sum(p.numel() for p in self.vision.parameters())/1e6:.2f}M",
                'evolution': f"{sum(p.numel() for p in self.evolution.parameters())/1e6:.2f}M"
            }
        }
print("✅ OpenMindAgent创建完成")
