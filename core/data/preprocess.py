from transformers import PreTrainedTokenizer

from typing import Dict, List

def pack_texts(element, max_length: int, text_column_name: str = "text") -> Dict[str, List[str]]:
    packed_texts = []
    current_batch = ""
    texts = element[text_column_name]

    while texts:
        for idx, entry in enumerate(texts):
            entry_length = len(entry)

            if len(current_batch) + entry_length > max_length:
                space_left = max_length - len(current_batch)
                current_batch += entry[:space_left]
                
                texts[idx] = entry[space_left:]

                if len(current_batch) == max_length:
                    packed_texts.append(current_batch)
                    current_batch = ""
                    break
            else:
                current_batch += entry
                texts[idx] = ""

        texts = [text for text in texts if text]

        if current_batch and not texts:
            packed_texts.append(current_batch)
            current_batch = ""

    return {"text": packed_texts}

def packed_tokenize(element, tokenizer: PreTrainedTokenizer) -> Dict[str, List[int]]:
    return tokenizer(element["text"], padding='max_length', truncation=True, return_tensors="pt")