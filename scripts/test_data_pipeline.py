"""
测试数据加载pipeline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import create_dataloader

def test_dataloader(data_path, description):
    print(f"\n{'='*60}")
    print(f"测试 {description}")
    print(f"{'='*60}")
    print(f"数据路径: {data_path}")
    
    try:
        dataloader = create_dataloader(
            data_path=data_path,
            tokenizer_name="Qwen/Qwen2-7B",
            batch_size=2,
            num_workers=0,
            shuffle=False,
            enable_audio=False
        )
        
        print("✅ 数据加载器创建成功")
        
        for i, batch in enumerate(dataloader):
            print(f"\n批次 {i+1}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  labels shape: {batch['labels'].shape}")
            
            if i == 0:
                print(f"\n示例文本（前50个token）:")
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
                text = tokenizer.decode(batch['input_ids'][0][:50])
                print(f"  {text}")
            
            if i >= 2:
                break
        
        print(f"\n✅ {description} 测试通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🧪 数据Pipeline测试")
    print("="*60)
    
    tests = [
        ("data/sample/pretrain/train.jsonl", "预训练数据加载"),
        ("data/sample/sft/train.jsonl", "SFT数据加载"),
    ]
    
    success_count = 0
    for data_path, description in tests:
        if Path(data_path).exists():
            if test_dataloader(data_path, description):
                success_count += 1
        else:
            print(f"\n⚠️  文件不存在: {data_path}")
            print("   请先运行: python scripts/create_sample_data.py")
    
    print("\n" + "="*60)
    if success_count == len(tests):
        print("✅ 所有数据加载测试通过！")
        print("\n📝 下一步:")
        print("  运行: python scripts/test_model_architecture.py")
    else:
        print(f"⚠️  {success_count}/{len(tests)} 测试通过")
    print("="*60)

if __name__ == "__main__":
    main()
