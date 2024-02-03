import os

def load_checkpoint(
    accelerator,
    train_dataloader,
    gradient_accumulation_steps: int = 1,
    resume_from_checkpoint: str = None
):
    if resume_from_checkpoint is not None or resume_from_checkpoint != "":
        accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")
        accelerator.load_state(resume_from_checkpoint)
        path = os.path.basename(resume_from_checkpoint)
    else:
        dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
        dirs.sort(key=os.path.getctime)
        path = dirs[-1]
    training_difference = os.path.splitext(path)[0]

    resume_step = int(training_difference.replace("step_", "")) * gradient_accumulation_steps
    starting_epoch = resume_step // len(train_dataloader)
    resume_step -= starting_epoch * len(train_dataloader)
    
    return accelerator, starting_epoch, resume_step

def save_checkpoint(
    accelerator,
    completed_steps: int,
    save_dir: str = "outputs/"
):
    step_dir = f"step_{completed_steps}"
    checkpoint_dir = os.path.join(save_dir, step_dir)
    accelerator.save_state(checkpoint_dir)
    accelerator.print(f"Checkpoint for step {completed_steps} saved at {checkpoint_dir}")

def save_unwrapped(
    accelerator,
    model,
    tokenizer,
    save_dir: str = "outputs/"
):
    checkpoint_dir = os.path.join(save_dir, f"completed")
    os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        checkpoint_dir, 
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(checkpoint_dir)
    accelerator.print(f"Checkpoint saved at {checkpoint_dir}")