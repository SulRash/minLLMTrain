import sentencepiece as spm

def get_spm_vocab_size(tokenizer_path: str):
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return sp.get_piece_size()
