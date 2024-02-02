import sentencepiece as spm

def get_spm_vocab_size(tokenizer_path: str):
    sp = spm.SentencePieceProcessor(model_file=model_file_path)
    return sp.get_piece_size()
