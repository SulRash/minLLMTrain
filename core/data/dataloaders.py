import os

import torch
import accelerate

from torch.utils.data.dataloader import DataLoader

from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset

from core.data.preprocess import *

from typing import Tuple

def get_datasets(
        accelerator,
        directory: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        text_field: str = "text"
) -> Tuple[Dataset, Dataset]:
    
    preprocessed_train_path = os.path.join(directory, 'preprocessed_train_dataset.pt')
    preprocessed_test_path = os.path.join(directory, 'preprocessed_test_dataset.pt')
    
    if os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_test_path):
        accelerator.print("Loading preprocessed datasets...")
        train_dataset = torch.load(preprocessed_train_path)
        test_dataset = torch.load(preprocessed_test_path)
    else:
        accelerator.print("Preprocessing datasets...")
        dataset = load_dataset('json', data_files=[os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jsonl')], split='train')
        dataset = dataset.map(lambda batch: tokenize(batch, tokenizer), remove_columns=dataset.column_names, batched=True)
        dataset = dataset.map(lambda batch: pack_tokenized_entries(batch, tokenizer, max_length), remove_columns=dataset.column_names, batched=True)
        dataset.set_format(type='torch')
        
        test_size = max(1, int(len(dataset) * 0.005))
        test_dataset = dataset.select(range(test_size))
        train_dataset = dataset.select(range(test_size, len(dataset)))
        
        torch.save(train_dataset, preprocessed_train_path)
        torch.save(test_dataset, preprocessed_test_path)
    
    return train_dataset, test_dataset

def get_dataloader(dataset: Dataset, batch_size: int = 16) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)