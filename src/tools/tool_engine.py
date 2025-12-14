import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from .base import BaseTool, ToolSpec, ToolResult, ToolRegistry, global_registry
from .builtin_tools import (
    CalculatorTool, CodeExecutorTool, 
    SearchTool, JSONParserTool, DateTimeTool
)
@dataclass
class ToolCall:
    """工具调用记录"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[ToolResult] = None
class ToolSelector(nn.Module):
    """工具选择器 - 根据输入选择合适的工具"""
    
    def __init__(self, hidden_size: int = 768, num_tools: int = 10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tools = num_tools
        
        # 工具选择网络
        self.tool_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_tools)
        )
        
        # 是否需要工具的判断
        self.need_tool = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 判断是否需要工具
        need_tool_prob = self.need_tool(hidden_states)
        
        # 选择工具
        tool_logits = self.tool_classifier(hidden_states)
        tool_probs = torch.softmax(tool_logits, dim=-1)
        
        return {
            'need_tool': need_tool_prob,
            'tool_probs': tool_probs,
            'selected_tool': tool_probs.argmax(dim=-1)
        }
class ArgumentExtractor(nn.Module):
    """参数提取器 - 从输入中提取工具参数"""
    
    def __init__(self, hidden_size: int = 768, max_args: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_args = max_args
        
        # 参数值提取
        self.arg_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * max_args)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 提取参数嵌入
        arg_embeddings = self.arg_extractor(hidden_states)
        arg_embeddings = arg_embeddings.view(-1, self.max_args, self.hidden_size)
        return arg_embeddings
class ToolEngine(nn.Module):
    """工具调用引擎 - 整合工具选择、参数提取和执行"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 初始化注册器并注册内置工具
        self.registry = ToolRegistry()
        self._register_builtin_tools()
        
        # 工具选择器
        self.selector = ToolSelector(
            hidden_size=hidden_size,
            num_tools=len(self.registry.tools) + 1  # +1 for "no tool"
        )
        
        # 参数提取器
        self.arg_extractor = ArgumentExtractor(hidden_size=hidden_size)
        
        # 结果编码器
        self.result_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 工具名称到索引的映射
        self.tool_names = ["none"] + list(self.registry.tools.keys())
        self.tool_to_idx = {name: idx for idx, name in enumerate(self.tool_names)}
        
        # 调用历史
        self.call_history: List[ToolCall] = []
        
    def _register_builtin_tools(self):
        """注册内置工具"""
        print("注册内置工具...")
        self.registry.register(CalculatorTool())
        self.registry.register(CodeExecutorTool())
        self.registry.register(SearchTool())
        self.registry.register(JSONParserTool())
        self.registry.register(DateTimeTool())
        
    def select_tool(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """选择工具"""
        selection = self.selector(hidden_states)
        
        # 获取选中的工具名称
        selected_idx = selection['selected_tool'].item() if selection['selected_tool'].dim() == 0 else selection['selected_tool'][0].item()
        selected_name = self.tool_names[selected_idx] if selected_idx < len(self.tool_names) else "none"
        
        return {
            'need_tool': selection['need_tool'].mean().item() > 0.5,
            'selected_tool': selected_name,
            'tool_probs': selection['tool_probs'],
            'confidence': selection['tool_probs'].max().item()
        }
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """执行工具"""
        result = self.registry.execute(tool_name, **kwargs)
        
        # 记录调用历史
        self.call_history.append(ToolCall(
            tool_name=tool_name,
            arguments=kwargs,
            result=result
        ))
        
        return result
    
    def forward(self, hidden_states: torch.Tensor,
                execute: bool = False,
                tool_args: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        工具引擎前向传播
        
        Args:
            hidden_states: 输入隐藏状态 [batch, hidden]
            execute: 是否执行选中的工具
            tool_args: 工具参数（如果execute=True）
        """
        results = {}
        
        # 选择工具
        selection = self.select_tool(hidden_states)
        results['selection'] = selection
        
        # 提取参数嵌入
        arg_embeddings = self.arg_extractor(hidden_states)
        results['arg_embeddings'] = arg_embeddings
        
        # 执行工具
        if execute and selection['need_tool'] and tool_args:
            tool_result = self.execute_tool(selection['selected_tool'], **tool_args)
            results['execution'] = {
                'tool': selection['selected_tool'],
                'result': tool_result
            }
            
            # 编码结果
            result_emb = self.result_encoder(hidden_states)
            results['result_embedding'] = result_emb
        
        # 统计
        results['stats'] = {
            'total_calls': len(self.call_history),
            'available_tools': self.registry.list_tools()
        }
        
        return results
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的schema"""
        return self.registry.get_all_specs()
print("✅ 工具调用引擎创建完成")
