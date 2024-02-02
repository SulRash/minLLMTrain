# What is This Repo

This repository is supposed to be a simplified repository for pretraining a large language model. It tries to leverage Deepspeed and Megatron-LM through Huggingface's Accelerate library for simplicity. It's still a work-in-progress. It currently implements a simplified version of dataset packing for improved efficiency during training.

# Using this Repo

After installing accelerate, run `accelerate config` and set-up accelerate correctly. I provided a simple deepspeed config in `configs/`, edit it to your needs.

Afterwards, run the bash script in the `examples/` directory to train a 1b parameter llama model after providing the directory in which jsonl files reside.