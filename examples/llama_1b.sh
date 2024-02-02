accelerate launch --config_file configs/bf16_z0_2gpu.yaml pretrain.py \
    --data_path "." \
    --config_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --tokenizer_path "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --save_dir "runs/"