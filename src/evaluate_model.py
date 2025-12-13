import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from typing import Dict, List
import numpy as np
from collections import defaultdict

from model_architecture import create_model
from data_pipeline import create_dataloader


class ModelEvaluator:
    def __init__(self, model_config: Dict, checkpoint_path: str):
        self.model = create_model(model_config)
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['base_model'])
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device('cpu')
    
    def load_checkpoint(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def evaluate_vqa(self, dataloader: DataLoader) -> Dict:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating VQA"):
                input_ids = batch['input_ids'].to(self.device)
                images = batch.get('images')
                if images is not None:
                    images = images.to(self.device)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    images=images,
                    max_length=50
                )
                
                predictions = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                labels = batch['labels'].to(self.device)
                label_texts = self.tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True
                )
                
                for pred, label in zip(predictions, label_texts):
                    if self._normalize_answer(pred) == self._normalize_answer(label):
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            'vqa_accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_reasoning(self, dataloader: DataLoader) -> Dict:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Reasoning"):
                input_ids = batch['input_ids'].to(self.device)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=512,
                    use_reasoning=True
                )
                
                predictions = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                labels = batch['labels'].to(self.device)
                label_texts = self.tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True
                )
                
                for pred, label in zip(predictions, label_texts):
                    if self._check_answer_match(pred, label):
                        correct += 1
                    total += 1
        
        accuracy = correct / total if total > 0 else 0
        return {
            'reasoning_accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_perplexity(self, dataloader: DataLoader) -> Dict:
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing Perplexity"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                num_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss
        }
    
    def evaluate_multimodal_reasoning(self, test_data_path: str) -> Dict:
        results = defaultdict(list)
        
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = [json.loads(line) for line in f]
        
        with torch.no_grad():
            for item in tqdm(test_data, desc="Multimodal Reasoning"):
                prompt = item.get('question', '')
                image_path = item.get('image')
                ground_truth = item.get('answer', '')
                task_type = item.get('type', 'general')
                
                input_ids = self.tokenizer(
                    prompt,
                    return_tensors='pt'
                ).input_ids.to(self.device)
                
                images = None
                if image_path:
                    from PIL import Image
                    from transformers import CLIPImageProcessor
                    
                    image_processor = CLIPImageProcessor.from_pretrained(
                        "openai/clip-vit-large-patch14"
                    )
                    image = Image.open(image_path).convert('RGB')
                    images = image_processor(
                        images=image,
                        return_tensors='pt'
                    )['pixel_values'].to(self.device)
                
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    images=images,
                    max_length=256,
                    use_reasoning=True
                )
                
                prediction = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
                
                is_correct = self._check_answer_match(prediction, ground_truth)
                results[task_type].append({
                    'prediction': prediction,
                    'ground_truth': ground_truth,
                    'correct': is_correct
                })
        
        summary = {}
        for task_type, items in results.items():
            correct = sum(item['correct'] for item in items)
            total = len(items)
            accuracy = correct / total if total > 0 else 0
            summary[f'{task_type}_accuracy'] = accuracy
        
        overall_correct = sum(
            sum(item['correct'] for item in items)
            for items in results.values()
        )
        overall_total = sum(len(items) for items in results.values())
        summary['overall_accuracy'] = overall_correct / overall_total if overall_total > 0 else 0
        
        return summary
    
    def _normalize_answer(self, text: str) -> str:
        text = text.lower().strip()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return ' '.join(text.split())
    
    def _check_answer_match(self, prediction: str, ground_truth: str) -> bool:
        pred_norm = self._normalize_answer(prediction)
        gt_norm = self._normalize_answer(ground_truth)
        
        if gt_norm in pred_norm:
            return True
        
        try:
            pred_num = float(''.join(c for c in pred_norm if c.isdigit() or c == '.'))
            gt_num = float(''.join(c for c in gt_norm if c.isdigit() or c == '.'))
            return abs(pred_num - gt_num) < 0.01
        except:
            pass
        
        return False
    
    def run_full_evaluation(self, eval_config: Dict) -> Dict:
        results = {}
        
        if 'vqa_data' in eval_config:
            print("\n=== Evaluating Visual Question Answering ===")
            vqa_loader = create_dataloader(
                data_path=eval_config['vqa_data'],
                tokenizer_name=self.model.config['base_model'],
                batch_size=8,
                shuffle=False
            )
            results['vqa'] = self.evaluate_vqa(vqa_loader)
        
        if 'reasoning_data' in eval_config:
            print("\n=== Evaluating Reasoning ===")
            reasoning_loader = create_dataloader(
                data_path=eval_config['reasoning_data'],
                tokenizer_name=self.model.config['base_model'],
                batch_size=4,
                shuffle=False
            )
            results['reasoning'] = self.evaluate_reasoning(reasoning_loader)
        
        if 'perplexity_data' in eval_config:
            print("\n=== Computing Perplexity ===")
            ppl_loader = create_dataloader(
                data_path=eval_config['perplexity_data'],
                tokenizer_name=self.model.config['base_model'],
                batch_size=8,
                shuffle=False
            )
            results['perplexity'] = self.evaluate_perplexity(ppl_loader)
        
        if 'multimodal_data' in eval_config:
            print("\n=== Evaluating Multimodal Reasoning ===")
            results['multimodal'] = self.evaluate_multimodal_reasoning(
                eval_config['multimodal_data']
            )
        
        return results


def main():
    model_config = {
        'base_model': 'Qwen/Qwen2-7B',
        'vision_model': 'openai/clip-vit-large-patch14',
        'freeze_vision': True,
        'perceiver_depth': 6,
        'num_latents': 64,
        'enable_audio': False,
        'enable_cot': True,
        'enable_verification': True
    }
    
    eval_config = {
        'vqa_data': 'data/eval_vqa.jsonl',
        'reasoning_data': 'data/eval_reasoning.jsonl',
        'perplexity_data': 'data/eval_ppl.jsonl',
        'multimodal_data': 'data/eval_multimodal.jsonl'
    }
    
    evaluator = ModelEvaluator(
        model_config=model_config,
        checkpoint_path='checkpoints/checkpoint-epoch-10/model.pt'
    )
    
    results = evaluator.run_full_evaluation(eval_config)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for task, metrics in results.items():
        print(f"\n{task.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    main()
