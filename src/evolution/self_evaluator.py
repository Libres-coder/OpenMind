import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
@dataclass
class EvaluationResult:
    """评估结果"""
    task_type: str
    score: float
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
class SelfEvaluator(nn.Module):
    """自我评估模块 - 评估模型输出质量"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 质量评估器
        self.quality_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 一致性评估器
        self.coherence_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 相关性评估器
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 综合评分器
        self.overall_scorer = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, output_embedding: torch.Tensor,
                input_embedding: torch.Tensor = None,
                context_embedding: torch.Tensor = None) -> Dict[str, Any]:
        """
        评估输出质量
        
        Args:
            output_embedding: 输出嵌入 [batch, hidden]
            input_embedding: 输入嵌入 [batch, hidden]
            context_embedding: 上下文嵌入 [batch, hidden]
        """
        batch_size = output_embedding.shape[0]
        
        # 质量分数
        quality = self.quality_scorer(output_embedding)
        
        # 一致性分数
        if input_embedding is not None:
            combined = torch.cat([output_embedding, input_embedding], dim=-1)
            coherence = self.coherence_scorer(combined)
        else:
            coherence = torch.ones(batch_size, 1, device=output_embedding.device) * 0.5
            
        # 相关性分数
        if context_embedding is not None:
            combined = torch.cat([output_embedding, context_embedding], dim=-1)
            relevance = self.relevance_scorer(combined)
        else:
            relevance = torch.ones(batch_size, 1, device=output_embedding.device) * 0.5
        
        # 综合分数
        scores = torch.cat([quality, coherence, relevance], dim=-1)
        overall = self.overall_scorer(scores)
        
        return {
            'quality_score': quality,
            'coherence_score': coherence,
            'relevance_score': relevance,
            'overall_score': overall,
            'needs_improvement': overall < 0.6
        }
print("✅ 自我评估模块创建完成")
