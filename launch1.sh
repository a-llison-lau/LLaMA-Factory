#!/bin/bash
#SBATCH --job-name=train_standard_cutofflen_2048
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=train_standard_cutoff2048.%A_%a.out
#SBATCH --error=train_standard_cutoff2048.%A_%a.err
#SBATCH --partition=a40
#SBATCH --qos=long
#SBATCH --time=2-00:00:00

conda_env="llm"
source activate $conda_env

# wandb agent allison-lau-university-of-toronto/LLaMA-Factory/7wt842kj
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml

