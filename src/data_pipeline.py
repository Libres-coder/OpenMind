import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, CLIPImageProcessor
import json
import os
from PIL import Image
import torchaudio
from typing import Dict, List, Optional, Tuple
import webdataset as wds


class MultimodalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_length: int = 2048,
        image_size: int = 224,
        enable_audio: bool = False
    ):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.max_length = max_length
        self.image_size = image_size
        self.enable_audio = enable_audio
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        text = item.get('text', '')
        conversation = item.get('conversations', [])
        
        if conversation:
            text = self._format_conversation(conversation)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }
        
        if 'image' in item:
            image = self._load_image(item['image'])
            result['images'] = image
            result['image_positions'] = [item.get('image_position', 0)]
        
        if self.enable_audio and 'audio' in item:
            audio = self._load_audio(item['audio'])
            result['audio'] = audio
        
        return result
    
    def _format_conversation(self, conversations: List[Dict]) -> str:
        formatted = []
        for turn in conversations:
            role = turn.get('from', 'user')
            content = turn.get('value', '')
            
            if role == 'human' or role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'gpt' or role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        return '\n'.join(formatted)
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        if image_path.startswith('http'):
            from io import BytesIO
            import requests
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            full_path = os.path.join(os.path.dirname(self.data_path), image_path)
            image = Image.open(full_path).convert('RGB')
        
        pixel_values = self.image_processor(
            images=image,
            return_tensors='pt'
        )['pixel_values'].squeeze(0)
        
        return pixel_values
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        full_path = os.path.join(os.path.dirname(self.data_path), audio_path)
        waveform, sample_rate = torchaudio.load(full_path)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        max_length = 16000 * 30
        if waveform.shape[1] > max_length:
            waveform = waveform[:, :max_length]
        
        return waveform.squeeze(0)


class WebDatasetLoader:
    def __init__(
        self,
        urls: List[str],
        tokenizer_name: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_buffer: int = 1000
    ):
        self.urls = urls
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer
    
    def create_dataset(self):
        dataset = (
            wds.WebDataset(self.urls, resampled=True)
            .shuffle(self.shuffle_buffer)
            .decode("rgb")
            .to_tuple("jpg", "json")
            .map(self._process_sample)
            .batched(self.batch_size)
        )
        return dataset
    
    def _process_sample(self, sample):
        image, metadata = sample
        
        text = metadata.get('caption', '')
        
        encoding = self.tokenizer(
            text,
            max_length=2048,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        pixel_values = self.image_processor(
            images=image,
            return_tensors='pt'
        )['pixel_values'].squeeze(0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'images': pixel_values,
            'labels': encoding['input_ids'].squeeze(0).clone()
        }
    
    def get_dataloader(self):
        dataset = self.create_dataset()
        return wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=self.num_workers
        )


def collate_fn(batch: List[Dict]) -> Dict:
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }
    
    if 'images' in batch[0]:
        images = torch.stack([item['images'] for item in batch])
        result['images'] = images
    
    if 'audio' in batch[0]:
        audio_list = [item['audio'] for item in batch]
        max_audio_len = max(a.shape[0] for a in audio_list)
        padded_audio = torch.zeros(len(audio_list), max_audio_len)
        for i, audio in enumerate(audio_list):
            padded_audio[i, :audio.shape[0]] = audio
        result['audio'] = padded_audio
    
    return result


def create_dataloader(
    data_path: str,
    tokenizer_name: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    enable_audio: bool = False
) -> DataLoader:
    dataset = MultimodalDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        enable_audio=enable_audio
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    dataloader = create_dataloader(
        data_path="data/train.jsonl",
        tokenizer_name="Qwen/Qwen2-7B",
        batch_size=4,
        num_workers=2
    )
    
    for batch in dataloader:
        print(f"Input shape: {batch['input_ids'].shape}")
        if 'images' in batch:
            print(f"Image shape: {batch['images'].shape}")
        break
