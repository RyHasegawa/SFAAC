#!/bin/bash

echo “start”
python batch_1_submit.py --data_dir=data/aac-unittest.jsonl --prompt_style=aac_z3 --model_name=gpt-4o
echo “Done!!!”
