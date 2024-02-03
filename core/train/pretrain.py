import torch

from tqdm import tqdm
from accelerate import DistributedType

from core.model.checkpoint import *

def evaluate(model, eval_dataloader, accelerator, per_device_batch_size):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
        loss = outputs.loss
        if accelerator.distributed_type == DistributedType.MEGATRON_LM:
            losses.append(loss)
        else:
            losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_batch_size)))
    try:
        if accelerator.distributed_type == DistributedType.MEGATRON_LM:
            losses = torch.tensor(losses)
        else:
            losses = torch.cat(losses)
        loss = torch.mean(losses)
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