import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .chain_of_thought import ChainOfThought
from .self_verification import SelfVerification
from .problem_decomposer import ProblemDecomposer
class ReasoningEngine(nn.Module):
    """推理引擎 - 整合所有推理能力"""
    
    def __init__(self, hidden_size: int = 768, max_cot_steps: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.cot = ChainOfThought(hidden_size, max_cot_steps)
        self.verifier = SelfVerification(hidden_size)
        self.decomposer = ProblemDecomposer(hidden_size)
        
        self.result_integrator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, hidden_states: torch.Tensor,
                problem_embedding: Optional[torch.Tensor] = None,
                use_decomposition: bool = True,
                use_verification: bool = True) -> Dict[str, Any]:
        results = {}
        
        if problem_embedding is None:
            problem_embedding = hidden_states.mean(dim=1)
            
        # 1. 问题分解
        if use_decomposition:
            decomp_result = self.decomposer(problem_embedding)
            results['decomposition'] = decomp_result
            
            # 修复: 使用.any()处理Tensor布尔值
            if decomp_result['should_decompose'].any():
                sub_results = []
                num_subs = min(decomp_result['num_subproblems'].max().item(), 3)
                for i in range(int(num_subs)):
                    sub_embed = decomp_result['subproblem_embeddings'][:, i, :]
                    sub_cot = self.cot(hidden_states, sub_embed)
                    sub_results.append(sub_cot)
                results['sub_reasoning'] = sub_results
        
        # 2. Chain-of-Thought推理
        cot_result = self.cot(hidden_states, problem_embedding)
        results['chain_of_thought'] = cot_result
        
        # 3. 自我验证
        if use_verification:
            verification = self.verifier(
                cot_result['final_state'],
                problem_embedding
            )
            results['verification'] = verification
            
            # 修复: 使用.any()处理Tensor布尔值
            if verification['needs_revision'].any():
                revised_cot = self.cot(hidden_states, cot_result['final_state'])
                results['revised_reasoning'] = revised_cot
                re_verification = self.verifier(
                    revised_cot['final_state'],
                    problem_embedding
                )
                results['re_verification'] = re_verification
        
        # 4. 整合结果
        final_state = cot_result['final_state']
        if 'revised_reasoning' in results:
            final_state = results['revised_reasoning']['final_state']
            
        combined = torch.cat([final_state, problem_embedding], dim=-1)
        results['final_output'] = self.result_integrator(combined)
        
        return results
print("✅ 推理引擎创建完成")
