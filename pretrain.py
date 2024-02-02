import torch

from tqdm import tqdm

from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

from core.model import *
from core.data.dataloaders import *
from core.utils.args import *

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
    num_training_steps: int, 
    gradient_accumulation_steps: int = 1,
    epochs: int = 1,
    eval_steps: int = 10,
    checkpoint_interval: float = 0.5,
    save_dir: str = "outputs/",
    exp_name: str = "first"
):

    model.train()
    
    completed_steps = 0
    checkpoint_step = int(num_training_steps * checkpoint_interval)  # Calculate the step for checkpointing

    for epoch in range(epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader), total=num_training_steps
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
            
            # Training
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                
            # Checkpointing
            if completed_steps % checkpoint_step == 0 and completed_steps != 0:
                checkpoint_dir = os.path.join(save_dir, f"{exp_name}-checkpoint-{completed_steps}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(checkpoint_dir)
                del unwrapped_model
                accelerator.print(f"Checkpoint saved at {checkpoint_dir}")
            
            # Evaluation
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                
 

def main():
    args = get_train_args()
    
    accelerator = Accelerator()
    
    config = get_config(args.config_path)
    tokenizer = get_tokenizer(args.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    model = get_model(config)

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
    
    train(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_training_steps=num_training_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        save_dir=args.save_dir,
        exp_name=args.exp_name
    )

main()