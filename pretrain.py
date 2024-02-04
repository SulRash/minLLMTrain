import math

from tqdm import tqdm

from torch.optim import AdamW
from accelerate import Accelerator, DistributedType
from accelerate.utils import MegatronLMDummyScheduler
from transformers import get_linear_schedule_with_warmup

from core.train.pretrain import train
from core.model.checkpoint import load_checkpoint
from core.model.load import *
from core.data.dataloaders import *
from core.args import *

def main():
    args = get_train_args()
    
    accelerator = Accelerator(project_dir=args.save_dir, gradient_accumulation_steps=args.gradient_accumulation_steps)    
    
    model, tokenizer, config = get_all_modelling(args.config_path, args.tokenizer_path)
    train_dataset, test_dataset = get_datasets(
        accelerator=accelerator
        directory=args.data_path,
        tokenizer=tokenizer,
        max_length=config.max_position_embeddings,
        text_field=args.text_field
    )
    train_dataloader, eval_dataloader = get_dataloader(train_dataset, args.per_device_batch_size), get_dataloader(test_dataset, args.per_device_batch_size)
    optimizer = AdamW(get_grouped_params(model), lr=args.lr)

    num_training_steps = (((len(train_dataloader) * args.epochs) // args.gradient_accumulation_steps) // accelerator.num_processes)    
    
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
    
    starting_epoch = 0
    resume_step = 0
    
    if args.resume_from_checkpoint:
        accelerator, starting_epoch, resume_step = load_checkpoint(
            accelerator=accelerator,
            train_dataloader=train_dataloader,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    progress_bar.update(starting_epoch * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    
    checkpoint_step = num_training_steps * args.checkpoint_interval
    eval_step = num_training_steps * args.eval_interval
    
    train(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        progress_bar=progress_bar,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        starting_epoch=starting_epoch,
        resume_step=resume_step,
        checkpoint_step=checkpoint_step,
        eval_step=eval_step,
        save_dir=args.save_dir
    )

main()