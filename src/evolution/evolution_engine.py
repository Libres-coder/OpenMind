import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from .self_evaluator import SelfEvaluator
from .experience_buffer import ExperienceBuffer, Experience
from .self_improver import SelfImprover
from datetime import datetime
class EvolutionEngine(nn.Module):
    """进化引擎 - 整合自我评估、经验回放和自我改进"""
    
    def __init__(self, hidden_size: int = 768,
                 buffer_capacity: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 核心组件
        self.evaluator = SelfEvaluator(hidden_size)
        self.improver = SelfImprover(hidden_size)
        self.experience_buffer = ExperienceBuffer(capacity=buffer_capacity)
        
        # 进化状态追踪
        self.evolution_step = 0
        self.improvement_history = []
        
        # 元学习器
        self.meta_learner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def evaluate_and_store(self, input_emb: torch.Tensor,
                           output_emb: torch.Tensor,
                           context_emb: torch.Tensor = None) -> Dict[str, Any]:
        """评估输出并存储经验"""
        # 评估
        eval_result = self.evaluator(output_emb, input_emb, context_emb)
        
        # 创建经验
        experience = Experience(
            input_data={'embedding': input_emb.detach().cpu().numpy().tolist()},
            output_data={'embedding': output_emb.detach().cpu().numpy().tolist()},
            reward=eval_result['overall_score'].mean().item(),
            evaluation={
                'quality': eval_result['quality_score'].mean().item(),
                'coherence': eval_result['coherence_score'].mean().item(),
                'relevance': eval_result['relevance_score'].mean().item()
            },
            timestamp=datetime.now().isoformat()
        )
        
        # 存储
        self.experience_buffer.add(experience)
        
        return {
            'evaluation': eval_result,
            'experience_stored': True,
            'buffer_size': len(self.experience_buffer)
        }
    
    def improve(self, output_emb: torch.Tensor,
                target_emb: torch.Tensor = None) -> Dict[str, Any]:
        """执行改进"""
        improve_result = self.improver(output_emb, target_emb)
        self.evolution_step += 1
        
        return {
            'improved_output': improve_result['improved_output'],
            'evolution_step': self.evolution_step
        }
    
    def forward(self, input_emb: torch.Tensor,
                output_emb: torch.Tensor,
                context_emb: torch.Tensor = None,
                target_emb: torch.Tensor = None,
                mode: str = 'full') -> Dict[str, Any]:
        """
        进化引擎前向传播
        
        Args:
            input_emb: 输入嵌入
            output_emb: 输出嵌入
            context_emb: 上下文嵌入
            target_emb: 目标嵌入（用于监督改进）
            mode: 'evaluate', 'improve', 'full'
        """
        results = {}
        
        if mode in ['evaluate', 'full']:
            eval_result = self.evaluate_and_store(input_emb, output_emb, context_emb)
            results.update(eval_result)
            
        if mode in ['improve', 'full']:
            improve_result = self.improve(output_emb, target_emb)
            results.update(improve_result)
            
        # 元学习更新
        if mode == 'full':
            combined = torch.cat([input_emb, output_emb], dim=-1)
            meta_output = self.meta_learner(combined)
            results['meta_representation'] = meta_output
            
        # 统计
        results['buffer_stats'] = self.experience_buffer.get_stats()
        
        return results
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """获取进化摘要"""
        return {
            'total_steps': self.evolution_step,
            'buffer_stats': self.experience_buffer.get_stats(),
            'high_quality_experiences': len(self.experience_buffer.get_high_quality()),
            'low_quality_experiences': len(self.experience_buffer.get_low_quality())
        }
print("✅ 进化引擎创建完成")
