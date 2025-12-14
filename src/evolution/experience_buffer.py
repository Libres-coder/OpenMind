import torch
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import random
import json
import os
@dataclass
class Experience:
    """单条经验"""
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reward: float
    evaluation: Dict[str, float]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
class ExperienceBuffer:
    """经验回放缓冲区 - 存储和采样历史经验"""
    
    def __init__(self, capacity: int = 10000, 
                 prioritized: bool = True):
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        
    def add(self, experience: Experience) -> None:
        """添加经验"""
        self.buffer.append(experience)
        # 优先级基于评估分数
        priority = experience.reward + 0.1  # 避免零优先级
        self.priorities.append(priority)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """采样经验批次"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        if self.prioritized:
            # 优先级采样
            total = sum(self.priorities)
            probs = [p / total for p in self.priorities]
            indices = random.choices(range(len(self.buffer)), 
                                    weights=probs, k=batch_size)
            return [self.buffer[i] for i in indices]
        else:
            return random.sample(list(self.buffer), batch_size)
    
    def get_high_quality(self, threshold: float = 0.7, 
                         top_k: int = 100) -> List[Experience]:
        """获取高质量经验"""
        high_quality = [e for e in self.buffer if e.reward >= threshold]
        high_quality.sort(key=lambda x: x.reward, reverse=True)
        return high_quality[:top_k]
    
    def get_low_quality(self, threshold: float = 0.4,
                        top_k: int = 100) -> List[Experience]:
        """获取需要改进的经验"""
        low_quality = [e for e in self.buffer if e.reward < threshold]
        low_quality.sort(key=lambda x: x.reward)
        return low_quality[:top_k]
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if len(self.buffer) == 0:
            return {'count': 0}
            
        rewards = [e.reward for e in self.buffer]
        return {
            'count': len(self.buffer),
            'avg_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'high_quality_count': len([r for r in rewards if r >= 0.7]),
            'low_quality_count': len([r for r in rewards if r < 0.4])
        }
print("✅ 经验回放模块创建完成")
