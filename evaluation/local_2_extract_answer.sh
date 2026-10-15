#!/bin/bash

echo “start”
python local_2_extract_answer.py --data_dir=data/aac-unittest-mini.jsonl --prompt_style=aac_z3 --model_name=Qwen/Qwen3-30B-A3B-Thinking-2507
echo “Done!!!”
