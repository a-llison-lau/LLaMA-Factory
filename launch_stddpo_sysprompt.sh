#!/bin/bash
#SBATCH --job-name=stddpo_sysprompt_sweep
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --output=stddpo_sysprompt_sweep.%j.out
#SBATCH --error=stddpo_sysprompt_sweep.%j.err
#SBATCH --account=def-rahulgk
#SBATCH --time=1-00
#SBATCH --mail-user=allison.lau@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023 intel/2023.2.1 arrow/15.0.1 cuda/11.8 python/3.10.13
source ~/vectorlm/bin/activate

export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN

# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Uncomment these flags for debugging communication
# export TORCH_CPP_LOG_LEVEL=INFO
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=0
export DISABLE_VERSION_CHECK=1

wandb offline

wandb agent allison-lau-university-of-toronto/LLaMA-Factory/7b5oj3lk