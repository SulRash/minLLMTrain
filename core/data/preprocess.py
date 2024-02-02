from transformers import PreTrainedTokenizer

from typing import Dict, List

def pack_tokenized_entries(tokenized_batch, tokenizer: PreTrainedTokenizer, max_length: int = 2048) -> Dict[str, List[int]]:
    packed_input_ids = []
    packed_attention_mask = []
    current_input_ids = []
    current_attention_mask = []

    for input_ids in tokenized_batch['input_ids']:
        if len(current_input_ids) + len(input_ids) + 1 > max_length:
            current_input_ids += [tokenizer.pad_token_id] * (max_length - len(current_input_ids))
            current_attention_mask += [0] * (max_length - len(current_attention_mask))
            
            packed_input_ids.append(current_input_ids)
            packed_attention_mask.append(current_attention_mask)
            
            current_input_ids = []
            current_attention_mask = []

        current_input_ids.extend(input_ids + [tokenizer.eos_token_id])
        current_attention_mask.extend([1] * len(input_ids) + [1])

    if current_input_ids:
        current_input_ids += [tokenizer.pad_token_id] * (max_length - len(current_input_ids))
        current_attention_mask += [0] * (max_length - len(current_attention_mask))
        packed_input_ids.append(current_input_ids)
        packed_attention_mask.append(current_attention_mask)

    return {
        'input_ids': packed_input_ids,
        'attention_mask': packed_attention_mask
    }

def tokenize(element, tokenizer: PreTrainedTokenizer, pad: bool = False) -> Dict[str, List[int]]:
    return tokenizer(element["text"], truncation=True, padding=pad)