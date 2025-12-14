import torch
from collections import deque
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
@dataclass
class MemoryItem:
    """单条记忆项"""
    role: str  # user / assistant / system
    content: str
    timestamp: datetime
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
class ShortTermMemory:
    """短期记忆 - 管理对话上下文窗口"""
    
    def __init__(self, max_turns: int = 10, max_tokens: int = 4096):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.memory: deque = deque(maxlen=max_turns * 2)  # user+assistant pairs
        
    def add(self, role: str, content: str, metadata: Dict = None) -> None:
        """添加一条记忆"""
        item = MemoryItem(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.memory.append(item)
        
    def get_context(self, last_n: int = None) -> List[MemoryItem]:
        """获取最近的对话上下文"""
        items = list(self.memory)
        if last_n:
            return items[-last_n:]
        return items
    
    def get_context_string(self, last_n: int = None) -> str:
        """获取格式化的上下文字符串"""
        items = self.get_context(last_n)
        context_parts = []
        for item in items:
            context_parts.append(f"[{item.role}]: {item.content}")
        return "\n".join(context_parts)
    
    def search(self, query: str, top_k: int = 3) -> List[MemoryItem]:
        """简单关键词搜索（后续可扩展为向量搜索）"""
        results = []
        query_lower = query.lower()
        for item in self.memory:
            if query_lower in item.content.lower():
                results.append(item)
        return results[:top_k]
    
    def clear(self) -> None:
        """清空短期记忆"""
        self.memory.clear()
        
    def __len__(self) -> int:
        return len(self.memory)
    
    def __repr__(self) -> str:
        return f"ShortTermMemory(turns={len(self)}/{self.max_turns*2})"
print("✅ 短期记忆模块创建完成")
