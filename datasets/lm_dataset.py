import json
import os
import torch
from torch.utils.data import Dataset

# 关闭 HuggingFace tokenizers 库的并行处理（多线程）功能。
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

"""
jsonl文件中 每一行包含一个完整的json对象
内存友好 方便解析
"""

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length-2,
            padding='max_length',
            truncation=True,  # 剪切超过max_length的文本
        ).input_ids

        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100  # 计算loss时忽略pad位置

        return input_ids, labels


    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)
        
        return samples