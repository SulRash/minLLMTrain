from transformers import AutoConfig, AutoModel, AutoTokenizer

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

def get_model(config_path: str, save_dir: str):
    """
    Instantiates a fresh model given a pre-existing model's config file.

    Args:
        config_path (str): Path to a huggingface config file.

    Returns:
        Huggingface model.
    """
    config = AutoConfig().from_pretrained(config_path)

    config.save_pretrained(save_dir)
    return AutoModel.from_config(config)

def get_tokenizer(tokenizer_path: str, save_dir: str):
    AutoTokenizer.from_pretrained