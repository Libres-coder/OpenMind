import torch
import torch.nn as nn
class QuantizedModel(nn.Module):
    """8-bit量化优化模型"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def quantize_weights(self):
        """量化模型权重到INT8"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # 计算scale和zero_point
                scale = (param.max() - param.min()) / 255
                zero_point = torch.round(-param.min() / scale)
                
                # 量化和反量化
                param_int8 = torch.round(param / scale + zero_point).clamp(0, 255)
                param.data = (param_int8 - zero_point) * scale
                
        print("✅ 模型量化完成")
        
    def get_model_size(self):
        """计算模型大小（MB）"""
        param_size = 0
        buffer_size = 0
        
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
print("✅ 量化模块创建完成")
