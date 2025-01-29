#!/bin/bash
#SBATCH --job-name=sysprompt
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=sysprompt.%A_%a.out
#SBATCH --error=sysprompt.%A_%a.err
#SBATCH --partition=a40
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --time=1-00:00:00

conda_env="llm"
source activate $conda_env

# wandb agent allison-lau-university-of-toronto/LLaMA-Factory/7wt842kj
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml

