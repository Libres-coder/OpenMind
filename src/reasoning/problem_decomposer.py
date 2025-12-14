import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
@dataclass
class SubProblem:
    """子问题"""
    id: int
    description: str
    complexity: float
    dependencies: List[int]
    solved: bool = False
    solution: Optional[str] = None
class ProblemDecomposer(nn.Module):
    """问题分解模块 - 将复杂问题拆解为子问题"""
    
    def __init__(self, hidden_size: int = 768, max_subproblems: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_subproblems = max_subproblems
        
        # 复杂度评估器
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 子问题生成器
        self.subproblem_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * max_subproblems)
        )
        
        # 依赖关系预测器
        self.dependency_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 子问题数量预测器
        self.num_predictor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, max_subproblems),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, problem_embedding: torch.Tensor) -> Dict[str, Any]:
        """
        分解问题
        
        Args:
            problem_embedding: 问题嵌入 [batch, hidden]
        """
        batch_size = problem_embedding.shape[0]
        
        # 1. 评估问题复杂度
        complexity = self.complexity_estimator(problem_embedding)
        
        # 2. 预测子问题数量
        num_probs = self.num_predictor(problem_embedding)
        predicted_num = num_probs.argmax(dim=-1) + 1  # 至少1个
        
        # 3. 生成子问题嵌入
        subproblem_embeds = self.subproblem_generator(problem_embedding)
        subproblem_embeds = subproblem_embeds.view(batch_size, self.max_subproblems, self.hidden_size)
        
        # 4. 预测依赖关系
        dependencies = []
        for i in range(self.max_subproblems):
            for j in range(i):
                combined = torch.cat([
                    subproblem_embeds[:, i, :],
                    subproblem_embeds[:, j, :]
                ], dim=-1)
                dep_prob = self.dependency_predictor(combined)
                if dep_prob.mean() > 0.5:
                    dependencies.append((i, j))
        
        return {
            'complexity': complexity,
            'num_subproblems': predicted_num,
            'subproblem_embeddings': subproblem_embeds,
            'dependencies': dependencies,
            'should_decompose': complexity > 0.5
        }
    
    def get_execution_order(self, dependencies: List[tuple]) -> List[int]:
        """获取子问题执行顺序（拓扑排序）"""
        from collections import defaultdict, deque
        
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        nodes = set()
        
        for i, j in dependencies:
            graph[j].append(i)  # j -> i (i depends on j)
            in_degree[i] += 1
            nodes.add(i)
            nodes.add(j)
            
        # BFS拓扑排序
        queue = deque([n for n in nodes if in_degree[n] == 0])
        order = []
        
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return order if len(order) == len(nodes) else list(range(self.max_subproblems))
print("✅ 问题分解模块创建完成")
