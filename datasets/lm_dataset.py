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
            max_length=self.max_length,
            padding='max_length',
            truncation=True,  # 剪切超过max_length的文本
            return_tensors='pt'
        )

        input_ids = tokens['input_ids'].squeeze(0)  # 由于是getitem 去掉batch维度
        loss_mask = input_ids != self.tokenizer.pad_token_id  # 只有非填充部分才计算损失

        X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入当前token
        Y = torch.tensor(input_ids[1:], dtype=torch.long)  # 预测下一个token
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 注意loss_mask要与Y对齐

        # X, Y -> [seq_len-1]
        return X, Y, loss_mask


    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                samples.append(data)
        
        return samples