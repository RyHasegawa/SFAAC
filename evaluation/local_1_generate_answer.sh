#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH -c 2
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a6000:1
#SBATCH --account=is-nlp
#SBATCH -o ./logs_slurm/slurm-_%j.out
#SBATCH --job-name=openmodel

echo “start”
python3 evaluation/local_1_generate_answer.py --data_dir=data/sfaac-1420.jsonl --prompt_style=aac_z3 --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507 --qwen3-only
echo “Done!!!”
