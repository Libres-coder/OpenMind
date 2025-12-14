import torch
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
@dataclass
class LongTermMemoryItem:
    """长期记忆项"""
    id: str
    content: str
    category: str  # fact / experience / skill / preference
    importance: float  # 0-1
    created_at: str
    last_accessed: str
    access_count: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
class LongTermMemory:
    """长期记忆 - 持久化存储重要信息"""
    
    def __init__(self, storage_path: str = "memory_store"):
        self.storage_path = storage_path
        self.memories: Dict[str, LongTermMemoryItem] = {}
        self.embeddings: Optional[torch.Tensor] = None
        self._ensure_storage()
        self._load()
        
    def _ensure_storage(self):
        """确保存储目录存在"""
        os.makedirs(self.storage_path, exist_ok=True)
        
    def _get_storage_file(self) -> str:
        return os.path.join(self.storage_path, "long_term_memory.json")
    
    def _load(self):
        """从文件加载记忆"""
        filepath = self._get_storage_file()
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item_data in data:
                    item = LongTermMemoryItem(**item_data)
                    self.memories[item.id] = item
            print(f"✅ 加载了 {len(self.memories)} 条长期记忆")
                    
    def save(self):
        """保存记忆到文件"""
        filepath = self._get_storage_file()
        data = [asdict(item) for item in self.memories.values()]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def add(self, content: str, category: str = "fact", 
            importance: float = 0.5, metadata: Dict = None) -> str:
        """添加长期记忆"""
        import uuid
        memory_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        
        item = LongTermMemoryItem(
            id=memory_id,
            content=content,
            category=category,
            importance=importance,
            created_at=now,
            last_accessed=now,
            access_count=0,
            metadata=metadata or {}
        )
        self.memories[memory_id] = item
        self.save()
        return memory_id
    
    def get(self, memory_id: str) -> Optional[LongTermMemoryItem]:
        """获取指定记忆"""
        item = self.memories.get(memory_id)
        if item:
            item.last_accessed = datetime.now().isoformat()
            item.access_count += 1
        return item
    
    def search(self, query: str, top_k: int = 5, 
               category: str = None) -> List[LongTermMemoryItem]:
        """搜索记忆（关键词匹配，后续可扩展为向量搜索）"""
        results = []
        query_lower = query.lower()
        
        for item in self.memories.values():
            if category and item.category != category:
                continue
            if query_lower in item.content.lower():
                results.append(item)
                
        # 按重要性排序
        results.sort(key=lambda x: x.importance, reverse=True)
        return results[:top_k]
    
    def forget(self, memory_id: str) -> bool:
        """删除记忆"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            self.save()
            return True
        return False
    
    def consolidate(self, min_importance: float = 0.3):
        """整理记忆 - 删除不重要的"""
        to_forget = [
            mid for mid, item in self.memories.items()
            if item.importance < min_importance and item.access_count < 2
        ]
        for mid in to_forget:
            self.forget(mid)
        return len(to_forget)
    
    def __len__(self) -> int:
        return len(self.memories)
print("✅ 长期记忆模块创建完成")
