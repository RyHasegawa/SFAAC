"""API Batchへ推論リクエストを投入する。"""

import argparse
import json
import os
import tomllib
import textwrap

from src.utils import (
    build_experiment_setting_name,
    ensure_parent_directory,
    filter_qwen3_records,
    read_jsonl,
)
from src.batch_api import add_to_batch_if_needed, execute_batch, load_hash_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--qwen3-only', action='store_true')
    args = parser.parse_args()

    prompt_config_path = "evaluation/config/prompts.toml"

    model_name = args.model_name
    prompt_style = args.prompt_style

    if ('CoT' in prompt_style) and ('gemini' not in model_name):
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "reasoning_and_answer",
            "schema": {
                "type": "object",
                "properties": {
                    "Reasoning": {"type": "string"},
                    "Answer": {"type": "string"}
                },
                "required": ["Reasoning", "Answer"],
                "additionalProperties": False
            },
            "strict": True
        }
        }

    elif ('prover9' in prompt_style) or ('Prover9' in prompt_style) or ('z3' in prompt_style) or ('Z3' in prompt_style):
        response_format = {'type': 'text'}
    else:
        response_format = {'type': 'json_object'}
    print(f'response_format: {response_format}')

    #tomlからプロンプト読み込み
    # prompt_style = args.prompt_style
    with open(prompt_config_path, mode="rb") as prompt_config_file:
        prompt_config_all= tomllib.load(prompt_config_file)
    #print(prompt_config_all)
    system_text   = textwrap.dedent(prompt_config_all[prompt_style]['system'])
    user_text     = textwrap.dedent(prompt_config_all[prompt_style]['user'])
    assistant_text = textwrap.dedent(prompt_config_all[prompt_style]['assistant'])
    #jsonデータ読み込み
    jsonl_data = filter_qwen3_records(read_jsonl(args.data_dir), args.qwen3_only)
    #print(jsonl_data)

    experiment_setting_name = build_experiment_setting_name(
        model_name, prompt_style, args.data_dir, args.qwen3_only
    )

	## キャッシュ
    file_name_cache = f'cache/cache_batch_{experiment_setting_name}.cache'
    if os.path.exists(file_name_cache):
    # ここにはハッシュ値と生データが格納されている
        print('cache existed')
        cache = load_hash_json(file_name_cache)
    else:
        cache = dict()
    ####

    #バッチ関連のファイル名
    batch_input_file = f"batch_input/batch_input_{experiment_setting_name}.jsonl" #モデルに投げるバッチファイルの名前
    batch_id_txt = f"batch_input/batch_id_{experiment_setting_name}.txt" # 確認用にbatchidを保存
    ensure_parent_directory(batch_input_file)
    ensure_parent_directory(file_name_cache)

    prompt_list = []
    for number, data in enumerate(jsonl_data):
        prompt = data['evaluation_text']
        prompt_list.append(prompt)

    batch_requests = []
    hash_list = []
    for t in prompt_list:
        hash_list.append(
            add_to_batch_if_needed(
                instruction=t,
                batch_requests=batch_requests,
                cache=cache,
                model_name=model_name,
                system_text=system_text,
                user_text=user_text,
                assistant_text=assistant_text,
                response_format=response_format
            ))

    execute_batch(batch_requests=batch_requests,
                  batch_input_file=batch_input_file,
                  batch_id_txt=batch_id_txt,
                  model_name=model_name)

    with open(f"batch_input/hash_list_{experiment_setting_name}.json", "w") as f:
        json.dump(hash_list, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
