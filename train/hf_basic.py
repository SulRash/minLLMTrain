from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def main():
    
    config = get_config()
    model = LlamaForCausalLM(config)
    model = torch.compile(model)
    
    tokenizer = LlamaTokenizerFast(vocab_file="model/tokenizer.model")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = get_dataloader(tokenizer, "")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=128)

    args = TrainingArguments(
        output_dir="model/",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=5_000,
        logging_steps=5_000,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=10_000,
        bf16=True
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset[-100:],
        eval_dataset=dataset[:-100]
    )
    
    trainer.train()