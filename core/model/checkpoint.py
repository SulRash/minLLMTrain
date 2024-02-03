import os

def checkpoint(
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