# Introduction

This repository is supposed to be a simplified repository for pretraining a large language model. It tries to leverage Deepspeed and Megatron-LM through Huggingface's Accelerate library for simplicity. It's still a work-in-progress. It currently implements a simplified version of dataset packing for improved efficiency during training.

## Using this Repo

I provided a simple accelerate config in `configs/`, edit it to your needs.

Afterwards, run the bash script in the `examples/` directory to train a 1b parameter llama model after providing the directory in which jsonl files reside.

# Training Frameworks

## Deepspeed

Deepspeed currently runs great! Check out the deepspeed configuration file in the `configs/` directory for an example.

## Megatron-LM

There's been some issues fully integrating Megatron-LM. Accelerate relies on (Huggingface's Megatron-LM repository)[https://github.com/huggingface/Megatron-LM]. The repository has long been abandoned, I am currently in the process of developing an updated fork that is compatible with Accelerate. In theory though, the training code here is designed to work with Megatron-LM.


## Fully Sharded Data Parallel

Should work, but it's a bit finicky. Support not mature yet.