from argparse import ArgumentParser, Namespace

def get_train_args() -> Namespace:
    
    parser = ArgumentParser()
    
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=float, default=0.5)
    parser.add_argument('--eval_interval', type=float, default=0.5)
    
    parser.add_argument('--per_device_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--save_dir', type=str)
    
    parser.add_argument('--text_field', type=str, default="text")
    
    return parser.parse_args()