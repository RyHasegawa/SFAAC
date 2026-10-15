#!/bin/bash

echo “start”
python3 evaluation/3_evaluate_answers.py --data_dir=data/sfaac-1420.jsonl --prompt_style=aac_z3 --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507 --qwen3-only
echo “Done!!!”
