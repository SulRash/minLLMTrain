import os

from torch.utils.data.dataloader import DataLoader

from transformers import PreTrainedTokenizer
from datasets import load_dataset, Dataset

from core.data.preprocess import *

from typing import Tuple

def get_datasets(
        directory: str,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 16,
        max_length: int = 2048,
        text_field: str = "text"
) -> Tuple[Dataset, Dataset]:
    dataset = load_dataset('json', data_files=[os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jsonl')], split='train')
    dataset = dataset.map(lambda batch: tokenize(batch, tokenizer), remove_columns=dataset.column_names, batched=True)
    dataset = dataset.map(lambda batch: pack_tokenized_entries(batch, tokenizer, max_length), remove_columns=dataset.column_names, batched=True)
    dataset.set_format(type='torch', columns=['input_ids'])

    test_size = max(1, int(len(dataset) * 0.01))
    test_dataset = dataset.select(range(test_size))
    train_dataset = dataset.select(range(test_size, len(dataset)))
    
    return train_dataset, test_dataset

def get_dataloader(dataset: Dataset, batch_size: int = 16) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)