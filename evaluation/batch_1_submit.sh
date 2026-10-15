#!/bin/bash

echo “start”
python3 evaluation/batch_1_submit.py --data_dir=data/sfaac-1420.jsonl --prompt_style=aac_z3 --model_name=gpt-4o
echo “Done!!!”
