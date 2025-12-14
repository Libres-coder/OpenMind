import uvicorn
import sys
sys.path.insert(0, '/root/autodl-tmp/OpenMind')
if __name__ == "__main__":
    print("🚀 启动OpenMind API服务...")
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
