import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPVisionModel
from typing import Optional, Dict, List, Tuple
import math

# 导入改进的视觉组件
from improved_vision_encoder import ImprovedVisionEncoder, ImprovedProjector


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 6,
        num_latents: int = 64,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim)
                )
            ]))
    
    def forward(self, x):
        batch_size = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        
        for ln1, attn, ln2, ff in self.layers:
            latents_norm = ln1(latents)
            attn_out, _ = attn(latents_norm, x, x)
            latents = latents + attn_out
            latents = latents + ff(ln2(latents))
        
        return latents


class VisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        freeze: bool = False
    ):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size
        
        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False
    
    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values, output_hidden_states=True)
        return outputs.last_hidden_state


class AudioEncoder(nn.Module):
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(512, hidden_size, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.hidden_size = hidden_size
    
    def forward(self, audio_input):
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(1)
        features = self.conv_layers(audio_input)
        features = features.transpose(1, 2)
        return features


class ReasoningHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        enable_cot: bool = True,
        enable_verification: bool = True
    ):
        super().__init__()
        self.enable_cot = enable_cot
        self.enable_verification = enable_verification
        
        self.cot_projection = nn.Linear(hidden_size, hidden_size)
        self.verification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states):
        if self.enable_cot:
            hidden_states = self.cot_projection(hidden_states)
        
        verification_scores = None
        if self.enable_verification:
            verification_scores = self.verification_head(hidden_states[:, -1, :])
        
        return hidden_states, verification_scores


class MultimodalReasoningModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # CPU环境：使用float32并启用低内存模式
        self.llm = AutoModel.from_pretrained(
            config['base_model'],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        
        # 使用改进的视觉编码器（SigLIP + Flash Attention）
        self.vision_encoder = ImprovedVisionEncoder(
            img_size=config.get('img_size', 384),
            patch_size=config.get('patch_size', 14),
            embed_dim=config.get('vision_embed_dim', 1152),
            depth=config.get('vision_depth', 27),
            num_heads=config.get('vision_heads', 16),
            use_flash_attn=config.get('use_flash_attn', True),
            dynamic_img_size=config.get('dynamic_img_size', True)
        )
        
        # 使用改进的投影器（Token Pooling，减少token数量）
        self.vision_projection = ImprovedProjector(
            input_dim=config.get('vision_embed_dim', 1152),
            output_dim=self.llm.config.hidden_size,
            projector_type=config.get('projector_type', 'token_pooling'),
            downsample_ratio=config.get('downsample_ratio', 2),
            depth=config.get('projector_depth', 2)
        )
        
        if config.get('enable_audio', False):
            self.audio_encoder = AudioEncoder(hidden_size=self.llm.config.hidden_size)
            self.audio_projection = nn.Linear(
                self.audio_encoder.hidden_size,
                self.llm.config.hidden_size
            )
        
        self.reasoning_head = ReasoningHead(
            hidden_size=self.llm.config.hidden_size,
            enable_cot=config.get('enable_cot', True),
            enable_verification=config.get('enable_verification', True)
        )
        
        self.lm_head = nn.Linear(
            self.llm.config.hidden_size,
            self.llm.config.vocab_size,
            bias=False
        )
        
        # CPU环境使用float32（移除bfloat16转换）
    
    def merge_multimodal_inputs(
        self,
        input_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """简化的多模态输入合并：视觉->文本->音频"""
        embeds_list = []
        
        # 顺序：视觉特征 + 文本 + 音频
        if vision_embeds is not None:
            embeds_list.append(vision_embeds)
        
        embeds_list.append(input_embeds)
        
        if audio_embeds is not None:
            embeds_list.append(audio_embeds)
        
        # 在序列维度(dim=1)拼接
        return torch.cat(embeds_list, dim=1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_positions: Optional[List[int]] = None,
        use_reasoning: bool = False
    ):
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        
        vision_embeds = None
        if images is not None:
            vision_features = self.vision_encoder(images)
            vision_embeds = self.vision_projection(vision_features)
        
        audio_embeds = None
        if audio is not None and hasattr(self, 'audio_encoder'):
            audio_features = self.audio_encoder(audio)
            audio_embeds = self.audio_projection(audio_features)
        
        combined_embeds = self.merge_multimodal_inputs(
            input_embeds,
            vision_embeds,
            audio_embeds
        )
        
        # 创建匹配的attention_mask
        batch_size, seq_len, _ = combined_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=combined_embeds.device)
        else:
            # 扩展attention_mask以匹配新的序列长度
            if vision_embeds is not None:
                vision_mask = torch.ones(batch_size, vision_embeds.shape[1], dtype=torch.long, device=combined_embeds.device)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
            if audio_embeds is not None:
                audio_mask = torch.ones(batch_size, audio_embeds.shape[1], dtype=torch.long, device=combined_embeds.device)
                attention_mask = torch.cat([attention_mask, audio_mask], dim=1)
        
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.last_hidden_state
        
        verification_scores = None
        if use_reasoning or self.training:
            hidden_states, verification_scores = self.reasoning_head(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            if verification_scores is not None:
                verification_loss = nn.BCELoss()(
                    verification_scores.squeeze(),
                    torch.ones_like(verification_scores.squeeze())
                )
                loss = loss + 0.1 * verification_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'verification_scores': verification_scores
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_reasoning: bool = False
    ):
        batch_size = input_ids.shape[0]
        
        for _ in range(max_length):
            outputs = self.forward(
                input_ids=input_ids,
                images=images,
                audio=audio,
                use_reasoning=use_reasoning
            )
            
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        return input_ids


def create_model(config: Dict) -> MultimodalReasoningModel:
    model = MultimodalReasoningModel(config)
    return model


if __name__ == "__main__":
    config = {
        'base_model': 'Qwen/Qwen2-7B',
        'vision_model': 'openai/clip-vit-large-patch14',
        'freeze_vision': True,
        'perceiver_depth': 6,
        'num_latents': 64,
        'enable_audio': True,
        'enable_cot': True,
        'enable_verification': True
    }
    
    model = create_model(config)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B")
