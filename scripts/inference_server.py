import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import logging
from src.optimized_model import OptimizedMultimodalModel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="OpenMind推理服务", version="1.0")
# 全局模型
model = None
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
class GenerateResponse(BaseModel):
    prompt: str
    generated_tokens: int
    status: str
@app.on_event("startup")
async def load_model():
    global model
    logger.info("加载模型...")
    config = {'hidden_size': 768, 'num_heads': 8, 'num_layers': 6}
    model = OptimizedMultimodalModel(config)
    model.eval()
    logger.info("✅ 模型加载完成")
@app.get("/")
async def root():
    return {"message": "OpenMind推理服务运行中", "status": "healthy"}
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    logger.info(f"收到请求: {request.prompt[:50]}...")
    
    # 模拟生成
    dummy_input = torch.randint(0, 50257, (1, 10))
    with torch.no_grad():
        output = model(dummy_input)
    
    return GenerateResponse(
        prompt=request.prompt,
        generated_tokens=request.max_tokens,
        status="success"
    )
if __name__ == "__main__":
    import uvicorn
    logger.info("启动推理服务器...")
    uvicorn.run(app, host="0.0.0.0", port=6006)
