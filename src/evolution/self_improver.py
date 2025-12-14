import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
class SelfImprover(nn.Module):
    """自我改进模块 - 基于反馈优化模型"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 错误分析器
        self.error_analyzer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 5)  # 5类错误
        )
        
        # 改进策略生成器
        self.strategy_generator = nn.Sequential(
            nn.Linear(hidden_size + 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 梯度调制器
        self.gradient_modulator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
    def analyze_error(self, output: torch.Tensor,
                      target: torch.Tensor) -> Dict[str, Any]:
        """分析错误类型"""
        combined = torch.cat([output, target], dim=-1)
        error_logits = self.error_analyzer(combined)
        error_probs = torch.softmax(error_logits, dim=-1)
        
        error_types = ['factual', 'logical', 'coherence', 'relevance', 'completeness']
        # 修复: 使用mean后再argmax，处理batch维度
        avg_error_probs = error_probs.mean(dim=0)
        dominant_idx = avg_error_probs.argmax().item()
        dominant_error = error_types[dominant_idx]
        
        return {
            'error_probs': error_probs,
            'dominant_error': dominant_error,
            'error_distribution': {t: avg_error_probs[i].item() 
                                   for i, t in enumerate(error_types)}
        }
    
    def generate_improvement(self, current_state: torch.Tensor,
                            error_info: Dict[str, Any]) -> torch.Tensor:
        """生成改进策略"""
        error_probs = error_info['error_probs']
        combined = torch.cat([current_state, error_probs], dim=-1)
        improvement = self.strategy_generator(combined)
        return improvement
    
    def forward(self, output: torch.Tensor,
                target: torch.Tensor = None,
                feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行自我改进
        
        Args:
            output: 当前输出 [batch, hidden]
            target: 目标输出 [batch, hidden]
            feedback: 外部反馈
        """
        results = {}
        
        # 如果有目标，分析错误
        if target is not None:
            error_info = self.analyze_error(output, target)
            results['error_analysis'] = error_info
            
            # 生成改进策略
            improvement = self.generate_improvement(output, error_info)
            results['improvement_vector'] = improvement
            
            # 计算梯度调制
            modulation = self.gradient_modulator(output)
            results['gradient_modulation'] = modulation
            
            # 改进后的输出
            results['improved_output'] = output + improvement * modulation
        else:
            # 无监督改进
            noise = torch.randn_like(output) * 0.1
            results['improved_output'] = output + noise
            
        return results
print("✅ 自我改进模块创建完成")
