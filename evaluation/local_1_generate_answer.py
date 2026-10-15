"""ローカルの公開済みモデルで回答を生成する。"""

import argparse
import json
import tomllib
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import (
    build_experiment_setting_name,
    ensure_parent_directory,
    filter_qwen3_records,
    read_jsonl,
)


def main():
    print("start")
    dt_now = datetime.datetime.now()
    print(dt_now)
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--qwen3-only', action='store_true')
    #parser.add_argument('--metrics_save_dir', default='./output', type=str)
    #parser.add_argument('--train_data_path', type=str)
    args = parser.parse_args()

    prompt_config_path = "evaluation/config/prompts.toml"

    model_name = args.model_name
    print(f'model_name: {model_name}')
    prompt_style = args.prompt_style
    print(f'prompt_style: {prompt_style}')

    #tomlからモデルの設定読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,     # 計算は bf16
        temperature=0.0
    )

    #tomlからプロンプト読み込み
    prompt_style = args.prompt_style
    with open(prompt_config_path, mode="rb") as prompt_config_file:
        prompt_config_all= tomllib.load(prompt_config_file)
    #print(prompt_config_all)
    system_text   = prompt_config_all[prompt_style]['system']
    user_text     = prompt_config_all[prompt_style]['user']
    assistant_text = prompt_config_all[prompt_style]['assistant']
    #jsonデータ読み込み
    jsonl_data = filter_qwen3_records(read_jsonl(args.data_dir), args.qwen3_only)
    #print(jsonl_data)

    experiment_setting_name = build_experiment_setting_name(
        model_name, prompt_style, args.data_dir, args.qwen3_only
    )

    # 出力JSON
    file_name_json = f'output_json/raw_output_{experiment_setting_name}.jsonl'
    ensure_parent_directory(file_name_json)

    output_json_list = []
    for number, data in enumerate(jsonl_data):
        # 生データ中の自然文: 'premise', 'conclusion'
        # LLMに生成させる論理式: 'Premise', 'Conclusion'
        # 似通っているので、生データ中の自然文を: 'premise_raw', 'conclusion_raw'に変更
        # LLMへの入力自然文は'prompt' = 'premise'+'conclusion'想定で問題なし
        # 'premise'+'conclusion'の処理はextract_data_aacorpus.pyで実施
        print(number)
        data['premise_raw'] = data.pop('premise')
        data['conclusion_raw'] = data.pop('conclusion')
        prompt = data['evaluation_text']
        messages = [{"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        try:
            generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=32768
            )
            generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            #print(response)
            output_json = {"raw_response": response}

        except:
            print(f"{number}: LLMerror")
            if 'CoT' in prompt_style:
                output_json = {"Reasoning": "LLMerror", "Answer": "LLMerror"}
            elif 'naive' in prompt_style:
                output_json = {"Answer": "LLMerror"}
            elif ('z3' in prompt_style) or ('Z3' in prompt_style) or ('prover9' in prompt_style) or ('Prover9' in prompt_style):
                output_json = {"raw_response": "LLMerror"}

        all_json = {**data, **output_json}
        output_json_list.append(all_json)

    # JSONL形式で保存
    with open(file_name_json, "w", encoding="utf-8") as f:
        for item in output_json_list:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    print("end")
    dt_now = datetime.datetime.now()
    print(dt_now)

if __name__ == "__main__":
    main()
