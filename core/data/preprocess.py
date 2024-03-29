from transformers import PreTrainedTokenizer

from collections import deque
from typing import Dict, List

def pack_tokenized_entries(tokenized_batch, tokenizer: PreTrainedTokenizer, max_length: int = 2048) -> Dict[str, List[int]]:
    packed_input_ids = []
    current_input_ids = deque()
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    for input_ids in tokenized_batch['input_ids']:
        input_length = len(input_ids) + 1

        if len(current_input_ids) + input_length > max_length:
            num_pads = max_length - len(current_input_ids)
            current_input_ids.extend([pad_token_id] * num_pads)

            packed_input_ids.append(list(current_input_ids))
            current_input_ids.clear()

        current_input_ids.extend(input_ids)
        current_input_ids.append(eos_token_id)

        if len(current_input_ids) > max_length:
            current_input_ids = deque(list(current_input_ids)[:max_length])

    if current_input_ids:
        num_pads = max_length - len(current_input_ids)
        current_input_ids.extend([pad_token_id] * num_pads)
        packed_input_ids.append(list(current_input_ids))

    return {
        'input_ids': packed_input_ids,
    }


def tokenize(element, tokenizer: PreTrainedTokenizer, text_field: str = "text", pad: bool = False) -> Dict[str, List[int]]:
    return tokenizer(element[text_field], truncation=True, padding=pad)
