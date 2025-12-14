import torch
from typing import List, Dict, Any, Optional
from .short_term_memory import ShortTermMemory, MemoryItem
from .long_term_memory import LongTermMemory, LongTermMemoryItem
class MemoryManager:
    """记忆管理器 - 统一管理短期和长期记忆"""
    
    def __init__(self, 
                 max_short_term_turns: int = 10,
                 storage_path: str = "memory_store"):
        self.short_term = ShortTermMemory(max_turns=max_short_term_turns)
        self.long_term = LongTermMemory(storage_path=storage_path)
        self.importance_threshold = 0.7  # 自动转为长期记忆的阈值
        
    def add_conversation(self, role: str, content: str, 
                         metadata: Dict = None) -> None:
        """添加对话到短期记忆"""
        self.short_term.add(role, content, metadata)
        
    def remember(self, content: str, category: str = "fact",
                 importance: float = 0.5) -> str:
        """显式添加到长期记忆"""
        return self.long_term.add(content, category, importance)
    
    def recall(self, query: str, include_short: bool = True,
               include_long: bool = True, top_k: int = 5) -> Dict[str, List]:
        """回忆相关记忆"""
        results = {
            'short_term': [],
            'long_term': []
        }
        
        if include_short:
            results['short_term'] = self.short_term.search(query, top_k)
            
        if include_long:
            results['long_term'] = self.long_term.search(query, top_k)
            
        return results
    
    def get_context_for_inference(self, query: str = None, 
                                   max_context_length: int = 2048) -> str:
        """获取推理用的上下文"""
        context_parts = []
        
        # 短期记忆（最近对话）
        short_context = self.short_term.get_context_string(last_n=6)
        if short_context:
            context_parts.append("## 最近对话\n" + short_context)
        
        # 长期记忆（相关知识）
        if query:
            relevant = self.long_term.search(query, top_k=3)
            if relevant:
                long_context = "\n".join([f"- {m.content}" for m in relevant])
                context_parts.append("## 相关知识\n" + long_context)
        
        return "\n\n".join(context_parts)
    
    def consolidate_to_long_term(self, content: str, importance: float) -> Optional[str]:
        """将重要内容转存到长期记忆"""
        if importance >= self.importance_threshold:
            return self.long_term.add(content, "experience", importance)
        return None
    
    def clear_short_term(self) -> None:
        """清空短期记忆"""
        self.short_term.clear()
        
    def save(self) -> None:
        """保存长期记忆"""
        self.long_term.save()
        
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            'short_term_count': len(self.short_term),
            'long_term_count': len(self.long_term),
            'categories': self._count_categories()
        }
    
    def _count_categories(self) -> Dict[str, int]:
        """统计各类别记忆数量"""
        counts = {}
        for item in self.long_term.memories.values():
            counts[item.category] = counts.get(item.category, 0) + 1
        return counts
print("✅ 记忆管理器创建完成")
