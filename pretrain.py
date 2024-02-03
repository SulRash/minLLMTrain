import torch

from tqdm import tqdm

from torch.optim import AdamW
from accelerate import Accelerator, DistributedType
from accelerate.utils import MegatronLMDummyScheduler
from transformers import get_linear_schedule_with_warmup

from core.train import train
from core.model.load import *
from core.data.dataloaders import *
from core.args import *

def main():
    args = get_train_args()
    
    accelerator = Accelerator(project_dir=args.save_dir, gradient_accumulation_steps=args.gradient_accumulation_steps)

    if accelerator.distributed_type == DistributedType.MEGATRON_LM:
        total_batch_size = accelerator.state.megatron_lm_plugin.global_batch_size
    else:
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    model, tokenizer, config = get_all_modelling(args.config_path, args.tokenizer_path)

    train_dataset, test_dataset = get_datasets(
        directory=args.data_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=config.max_position_embeddings,
        text_field=args.text_field
    )
    
    train_dataloader, eval_dataloader = get_dataloader(train_dataset, args.batch_size), get_dataloader(test_dataset, args.batch_size)
    num_training_steps = (len(train_dataloader) * args.epochs) // args.gradient_accumulation_steps
    
    optimizer = AdamW(get_grouped_params(model), lr=args.lr)
    
    if accelerator.distributed_type == DistributedType.MEGATRON_LM:
        scheduler = MegatronLMDummyScheduler(
            optimizer=optimizer,
            total_num_steps=num_training_steps,
            warmup_num_steps=args.num_warmup_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    model, optimizer, train_dataloader, scheduler, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler, eval_dataloader
    )
    
    accelerator.register_for_checkpointing(scheduler)
        
    train(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.save_dir
    )

main()