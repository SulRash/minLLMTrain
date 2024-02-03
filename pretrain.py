import torch

from tqdm import tqdm

from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

from core.model.checkpoint import *
from core.model.load import *
from core.data.dataloaders import *
from core.args import *

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

def train( 
    accelerator, 
    model, 
    tokenizer, 
    train_dataloader, 
    eval_dataloader, 
    optimizer, 
    scheduler, 
    gradient_accumulation_steps: int = 1,
    epochs: int = 1,
    eval_steps: int = 10,
    checkpoint_interval: float = 0.5,
    save_dir: str = "outputs/"
):

    model.train()
    
    completed_steps = 0
    checkpoint_step = int(len(train_dataloader) * epochs * checkpoint_interval)

    for epoch in range(epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            loss = model(input_ids=batch['input_ids'], labels=batch['input_ids'], attention_mask=batch['attention_mask']).loss
            
            if step % 100 == 0:
                accelerator.print(
                    {
                        "steps": completed_steps,
                        "loss/train": loss * gradient_accumulation_steps,
                    }
                )
            
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            
            # Stepping
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                
            # Checkpointing
            if completed_steps % checkpoint_step == 0 and completed_steps != 0:
                checkpoint(
                    accelerator, completed_steps, save_dir
                )
            
            # Evaluation
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
    
    # Save final model as HF model
    save_unwrapped(
        accelerator, model, tokenizer, save_dir
    )

def main():
    args = get_train_args()
    
    accelerator = Accelerator(project_dir=args.save_dir)
    
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
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
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