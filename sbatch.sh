#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode27
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2025-Apr-10/abdulbasit/training_llm_output.out
#SBATCH --error=/scratch/expires-2025-Apr-10/abdulbasit/training_llm_output.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 run.py --config configs/train_llama.json --constraint_type all --run_name super-loco-llama-3.2-3b