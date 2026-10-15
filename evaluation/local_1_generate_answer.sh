#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH -c 2
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=is-nlp
#SBATCH -o ./logs_slurm/slurm-_%j.out
#SBATCH --job-name=openmodel

echo “start”
python local_1_generate_answer.py --data_dir=data/aac-unittest-mini.jsonl --prompt_style=aac_z3 --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507
echo “Done!!!”
