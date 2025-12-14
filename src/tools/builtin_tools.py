import math
import json
import subprocess
from typing import Dict, Any, List, Optional
from .base import BaseTool, ToolSpec, ToolResult
class CalculatorTool(BaseTool):
    """数学计算工具"""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算，支持基本运算和数学函数"
        )
    
    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2+3*4' 或 'sqrt(16)'"
                }
            },
            required=["expression"]
        )
    
    def execute(self, expression: str) -> ToolResult:
        try:
            # 安全的数学环境
            safe_dict = {
                "abs": abs, "round": round,
                "min": min, "max": max,
                "sum": sum, "pow": pow,
                "sqrt": math.sqrt, "log": math.log,
                "sin": math.sin, "cos": math.cos,
                "tan": math.tan, "pi": math.pi,
                "e": math.e, "exp": math.exp
            }
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
class CodeExecutorTool(BaseTool):
    """Python代码执行工具"""
    
    def __init__(self, timeout: int = 10):
        super().__init__(
            name="code_executor",
            description="执行Python代码片段并返回结果"
        )
        self.timeout = timeout
    
    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "code": {
                    "type": "string",
                    "description": "要执行的Python代码"
                }
            },
            required=["code"]
        )
    
    def execute(self, code: str) -> ToolResult:
        try:
            # 创建隔离的执行环境
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            
            # 获取result变量或最后一个表达式的值
            output = local_vars.get('result', local_vars.get('output', str(local_vars)))
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
class SearchTool(BaseTool):
    """搜索工具（模拟）"""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="搜索互联网获取信息"
        )
        # 模拟的知识库
        self.knowledge_base = {
            "python": "Python是一种高级编程语言，以其简洁和可读性著称。",
            "pytorch": "PyTorch是一个开源机器学习框架，由Facebook开发。",
            "transformer": "Transformer是一种基于自注意力机制的神经网络架构。",
            "gpt": "GPT是Generative Pre-trained Transformer的缩写，是一种大语言模型。",
            "多模态": "多模态AI能够处理和理解多种类型的数据，如文本、图像、音频等。"
        }
    
    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                }
            },
            required=["query"]
        )
    
    def execute(self, query: str) -> ToolResult:
        query_lower = query.lower()
        results = []
        for key, value in self.knowledge_base.items():
            if key in query_lower or query_lower in key:
                results.append({"topic": key, "content": value})
        
        if results:
            return ToolResult(success=True, output=results)
        else:
            return ToolResult(
                success=True, 
                output=[{"topic": "未找到", "content": f"没有找到关于'{query}'的信息"}]
            )
class JSONParserTool(BaseTool):
    """JSON解析工具"""
    
    def __init__(self):
        super().__init__(
            name="json_parser",
            description="解析和处理JSON数据"
        )
    
    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "json_string": {
                    "type": "string",
                    "description": "JSON字符串"
                },
                "path": {
                    "type": "string",
                    "description": "可选的JSON路径，如 'data.items[0].name'"
                }
            },
            required=["json_string"]
        )
    
    def execute(self, json_string: str, path: str = None) -> ToolResult:
        try:
            data = json.loads(json_string)
            
            if path:
                # 简单的路径解析
                parts = path.replace('[', '.').replace(']', '').split('.')
                result = data
                for part in parts:
                    if part.isdigit():
                        result = result[int(part)]
                    else:
                        result = result[part]
                return ToolResult(success=True, output=result)
            
            return ToolResult(success=True, output=data)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
class DateTimeTool(BaseTool):
    """日期时间工具"""
    
    def __init__(self):
        super().__init__(
            name="datetime",
            description="获取当前日期时间或进行日期计算"
        )
    
    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "operation": {
                    "type": "string",
                    "enum": ["now", "format", "diff"],
                    "description": "操作类型"
                },
                "format": {
                    "type": "string",
                    "description": "日期格式，如 '%Y-%m-%d'"
                }
            },
            required=["operation"]
        )
    
    def execute(self, operation: str, format: str = "%Y-%m-%d %H:%M:%S") -> ToolResult:
        from datetime import datetime
        try:
            if operation == "now":
                result = datetime.now().strftime(format)
                return ToolResult(success=True, output=result)
            elif operation == "format":
                return ToolResult(success=True, output=format)
            else:
                return ToolResult(success=False, output=None, error=f"Unknown operation: {operation}")
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
print("✅ 内置工具创建完成")
