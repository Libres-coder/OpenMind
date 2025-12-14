import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
@dataclass
class ThoughtStep:
    """思考步骤"""
    step_id: int
    thought: str
    confidence: float
    reasoning_type: str  # analysis / deduction / verification
class ChainOfThought(nn.Module):
    """Chain-of-Thought推理模块"""
    
    def __init__(self, hidden_size: int = 768, max_steps: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        
        # 思考步骤生成器
        self.thought_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # 步骤控制器（决定是否继续推理）
        self.step_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 置信度评估器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor, 
                problem_embedding: torch.Tensor = None) -> Dict[str, Any]:
        """
        执行Chain-of-Thought推理
        
        Args:
            hidden_states: 输入隐藏状态 [batch, seq, hidden]
            problem_embedding: 问题嵌入 [batch, hidden]
        """
        batch_size = hidden_states.shape[0]
        
        # 初始化
        if problem_embedding is None:
            current_state = hidden_states.mean(dim=1)  # [batch, hidden]
        else:
            current_state = problem_embedding
            
        thought_chain = []
        all_states = [current_state]
        
        for step in range(self.max_steps):
            # 生成下一个思考步骤
            next_thought = self.thought_generator(current_state)
            
            # 评估置信度
            confidence = self.confidence_estimator(next_thought)
            
            # 决定是否继续
            continue_prob = self.step_controller(next_thought)
            
            thought_chain.append({
                'step': step,
                'state': next_thought,
                'confidence': confidence.item() if batch_size == 1 else confidence,
                'continue_prob': continue_prob.item() if batch_size == 1 else continue_prob
            })
            
            # 更新状态
            current_state = next_thought
            all_states.append(current_state)
            
            # 如果置信度足够高，提前终止
            if confidence.mean() > 0.9:
                break
                
        # 聚合所有思考步骤
        final_state = torch.stack(all_states, dim=1).mean(dim=1)
        
        return {
            'final_state': final_state,
            'thought_chain': thought_chain,
            'num_steps': len(thought_chain),
            'final_confidence': thought_chain[-1]['confidence']
        }
    
    def format_thought_chain(self, thought_chain: List[Dict]) -> str:
        """格式化思考链为可读文本"""
        lines = ["思考过程:"]
        for t in thought_chain:
            conf = t['confidence']
            if isinstance(conf, torch.Tensor):
                conf = conf.mean().item()
            lines.append(f"  步骤{t['step']+1}: 置信度={conf:.2f}")
        return "\n".join(lines)
print("✅ Chain-of-Thought推理模块创建完成")
