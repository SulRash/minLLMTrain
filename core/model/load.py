from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast

from typing import Dict, List

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]) -> List[Dict]:
    weight_decay = 0.1
    
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def get_config(config_path: str):
    config = AutoConfig.from_pretrained(config_path)
    return config

def get_model(config):
    return AutoModelForCausalLM.from_config(config)

def get_tokenizer(tokenizer_path: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def get_tokenizer_from_file(tokenizer_path: str):
    tokenizer = LlamaTokenizerFast(
        vocab_file=tokenizer_path
    )
    return tokenizer

def get_all_modelling(config_path, tokenizer_path):
    config = get_config(config_path)
    tokenizer = get_tokenizer(tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    model = get_model(config)
    
    return model, tokenizer, config