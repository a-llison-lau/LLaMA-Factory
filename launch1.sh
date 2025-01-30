#!/bin/bash
#SBATCH --job-name=sweep_sysprompt_standard
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=sweep_sysprompt_standard.%A_%a.out
#SBATCH --error=sweep_sysprompt_standard.%A_%a.err
#SBATCH --partition=a40
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --time=1-00:00:00

conda_env="llm"
source activate $conda_env

# wandb agent allison-lau-university-of-toronto/LLaMA-Factory/7wt842kj
# llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
wandb agent allison-lau-university-of-toronto/LLaMA-Factory/m3syywy3
