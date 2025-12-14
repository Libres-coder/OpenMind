import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
@dataclass
class VerificationResult:
    """验证结果"""
    is_valid: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]
class SelfVerification(nn.Module):
    """自我验证模块 - 检查推理结果的正确性"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 一致性检查器
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 逻辑验证器
        self.logic_validator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 完整性检查器
        self.completeness_checker = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 综合评估器
        self.overall_evaluator = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, reasoning_output: torch.Tensor,
                original_input: torch.Tensor = None) -> Dict[str, Any]:
        """
        验证推理结果
        
        Args:
            reasoning_output: 推理输出 [batch, hidden]
            original_input: 原始输入 [batch, hidden]
        """
        batch_size = reasoning_output.shape[0]
        
        # 1. 一致性检查（输入输出一致性）
        if original_input is not None:
            combined = torch.cat([reasoning_output, original_input], dim=-1)
            consistency_score = self.consistency_checker(combined)
        else:
            consistency_score = torch.ones(batch_size, 1, device=reasoning_output.device) * 0.5
            
        # 2. 逻辑验证
        logic_score = self.logic_validator(reasoning_output)
        
        # 3. 完整性检查
        completeness_score = self.completeness_checker(reasoning_output)
        
        # 4. 综合评估
        scores = torch.cat([consistency_score, logic_score, completeness_score], dim=-1)
        overall_score = self.overall_evaluator(scores)
        
        return {
            'is_valid': overall_score > 0.5,
            'overall_score': overall_score,
            'consistency_score': consistency_score,
            'logic_score': logic_score,
            'completeness_score': completeness_score,
            'needs_revision': overall_score < 0.7
        }
    
    def get_feedback(self, verification_result: Dict[str, Any]) -> str:
        """生成验证反馈"""
        feedback = []
        
        if verification_result['consistency_score'].mean() < 0.6:
            feedback.append("⚠️ 一致性不足：结论与前提可能不匹配")
        if verification_result['logic_score'].mean() < 0.6:
            feedback.append("⚠️ 逻辑问题：推理过程可能存在漏洞")
        if verification_result['completeness_score'].mean() < 0.6:
            feedback.append("⚠️ 不完整：可能遗漏了重要信息")
            
        if not feedback:
            feedback.append("✅ 验证通过：推理结果可靠")
            
        return "\n".join(feedback)
print("✅ 自我验证模块创建完成")
