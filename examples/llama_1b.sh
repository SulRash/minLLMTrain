accelerate launch --config_file configs/deepspeed.yaml pretrain.py \
    --epochs 5 \
    --data_path "." \
    --config_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --tokenizer_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --save_dir "runs/"