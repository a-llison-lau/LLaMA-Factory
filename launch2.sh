#!/bin/bash
#SBATCH --job-name=sweep_standard
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=sweep_standard.%A_%a.out
#SBATCH --error=sweep_standard.%A_%a.err
#SBATCH --partition=a40
#SBATCH --qos=long
#SBATCH --time=2-00:00:00

conda_env="llm"
source activate $conda_env

# wandb agent allison-lau-university-of-toronto/LLaMA-Factory/ytgfkmye
# llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
wandb agent allison-lau-university-of-toronto/LLaMA-Factory/tgfycusz

