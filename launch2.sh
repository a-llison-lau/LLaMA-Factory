#!/bin/bash
#SBATCH --job-name=sweep_sysprompt
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=sweep_sysprompt.%A_%a.out
#SBATCH --error=sweep_sysprompt.%A_%a.err
#SBATCH --partition=a40
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --time=1-00:00:00

conda_env="llm"
source activate $conda_env

# wandb agent allison-lau-university-of-toronto/LLaMA-Factory/ytgfkmye
# llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
wandb agent allison-lau-university-of-toronto/LLaMA-Factory/qg4e18uf

