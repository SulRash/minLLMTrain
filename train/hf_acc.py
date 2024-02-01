import torch

from tqdm import tqdm

from torch.optim import AdamW

from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

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
    accelerator, model, epochs, train_dataloader, num_training_steps, optimizer, lr_scheduler, tokenizer, eval_dataloader = None, eval_steps: int = 10, output_dir: str = "outputs/"
):
    gradient_accumulation_steps = 1

    model.train()
    completed_steps = 0
    for epoch in range(epochs):
        for step, batch in tqdm(
            enumerate(train_dataloader), total=num_training_steps
        ):
            loss = model(batch["input_ids"]).loss
            if step % 100 == 0:
                accelerator.print(
                    {
                        "steps": completed_steps,
                        "loss/train": loss * gradient_accumulation_steps,
                    }
                )
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            # if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            #     eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)
            #     accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            #     model.train()
            #     accelerator.wait_for_everyone()
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            #     if accelerator.is_main_process:
            #         tokenizer.save_pretrained(output_dir)
 

def main():
    epochs = 1
    gradient_accumulation_steps = 1
    lr = 5e-4
    
    accelerator = Accelerator()
    
    config = get_config()
    model = LlamaForCausalLM(config)
    
    tokenizer = LlamaTokenizerFast(vocab_file="model/tokenizer.model")
    tokenizer.pad_token = tokenizer.eos_token

    train_dataloader, _ = get_dataloaders(tokenizer, "")
    
    num_training_steps = (len(train_dataloader) * epochs) // gradient_accumulation_steps
    
    optimizer = AdamW(get_grouped_params(model), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train(
        model=model,
        accelerator=accelerator,
        epochs=epochs,
        train_dataloader=train_dataloader,
        num_training_steps=num_training_steps,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        tokenizer=tokenizer,
    )