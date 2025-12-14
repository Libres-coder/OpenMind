import torch
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import base64
import io
from PIL import Image
import numpy as np
# 导入Agent
import sys
sys.path.insert(0, '/root/autodl-tmp/OpenMind')
from src.core import OpenMindAgent, AgentConfig
# 创建FastAPI应用
app = FastAPI(
    title="OpenMind API",
    description="多模态智能Agent API服务",
    version="1.0.0"
)
# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 全局Agent实例
agent = None
device = None
# 请求模型
class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None
    use_reasoning: bool = True
    use_evolution: bool = True
class ChatResponse(BaseModel):
    status: str
    mode: str
    output_shape: List[int]
    reasoning_steps: Optional[int] = None
    evolution_score: Optional[float] = None
    memory_context_length: Optional[int] = None
class AnalyzeImageRequest(BaseModel):
    image_base64: str
    task: str = "classify"  # classify, caption, encode
class StatsResponse(BaseModel):
    total_parameters: str
    trainable_parameters: str
    components: Dict[str, str]
    memory_stats: Dict[str, Any]
    evolution_stats: Dict[str, Any]
@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global agent, device
    
    print("🚀 正在加载OpenMind Agent...")
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 使用设备: {device}")
    
    # 创建配置
    config = AgentConfig(
        hidden_size=768,
        max_cot_steps=5,
        img_size=224,
        vision_layers=6,
        fusion_layers=4
    )
    
    # 创建Agent
    agent = OpenMindAgent(config)
    agent = agent.to(device)
    agent.eval()
    
    print(f"✅ Agent加载完成! 参数量: {sum(p.numel() for p in agent.parameters())/1e6:.2f}M")
@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "OpenMind API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/chat", "/analyze", "/stats", "/health"]
    }
@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "agent_loaded": agent is not None,
        "device": str(device)
    }
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """获取Agent统计信息"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent未加载")
    
    stats = agent.get_stats()
    return StatsResponse(**stats)
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent未加载")
    
    try:
        # 创建文本嵌入 (模拟，实际应该用tokenizer)
        text_emb = torch.randn(1, 768).to(device)
        
        # 处理图像
        image_tensor = None
        if request.image_base64:
            # 解码base64图像
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = image.resize((224, 224))
            image_np = np.array(image) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # 调用Agent
        with torch.no_grad():
            if image_tensor is not None:
                result = agent.chat(request.message, text_emb, image_tensor)
                mode = "multimodal"
            else:
                result = agent.chat(request.message, text_emb)
                mode = "text"
        
        # 构建响应
        response = ChatResponse(
            status="success",
            mode=mode,
            output_shape=list(result['output'].shape),
            reasoning_steps=result.get('reasoning', {}).get('chain_of_thought', {}).get('num_steps'),
            evolution_score=result.get('evolution', {}).get('evaluation', {}).get('overall_score', torch.tensor(0)).mean().item() if 'evolution' in result else None,
            memory_context_length=len(result.get('memory_context', ''))
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/analyze")
async def analyze_image(request: AnalyzeImageRequest):
    """图像分析接口"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent未加载")
    
    try:
        # 解码图像
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))
        image_np = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # 调用视觉系统
        with torch.no_grad():
            result = agent.vision(image_tensor, task=request.task)
        
        response = {
            "status": "success",
            "task": request.task
        }
        
        if request.task == "classify":
            logits = result.get('logits')
            if logits is not None:
                top_k = torch.topk(logits, 5, dim=-1)
                response["top_5_classes"] = top_k.indices[0].tolist()
                response["top_5_scores"] = top_k.values[0].tolist()
        elif request.task == "encode":
            response["cls_token_shape"] = list(result.get('vision_cls', torch.zeros(1)).shape)
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
print("✅ API服务模块创建完成")
