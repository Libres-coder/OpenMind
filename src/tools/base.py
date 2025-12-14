import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import json
@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class ToolSpec:
    """工具规格定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)
class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._spec: Optional[ToolSpec] = None
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass
    
    @abstractmethod
    def get_spec(self) -> ToolSpec:
        """获取工具规格"""
        pass
    
    def to_function_schema(self) -> Dict[str, Any]:
        """转换为OpenAI风格的function schema"""
        spec = self.get_spec()
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": spec.parameters,
                "required": spec.required
            }
        }
class ToolRegistry:
    """工具注册器"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool) -> None:
        """注册工具"""
        self.tools[tool.name] = tool
        print(f"  ✓ 注册工具: {tool.name}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self.tools.keys())
    
    def get_all_specs(self) -> List[Dict[str, Any]]:
        """获取所有工具的schema"""
        return [tool.to_function_schema() for tool in self.tools.values()]
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """执行指定工具"""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found"
            )
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
# 全局注册器
global_registry = ToolRegistry()
print("✅ 工具基类创建完成")
