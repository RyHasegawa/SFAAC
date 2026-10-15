import argparse
import json
import re

import pandas as pd

from src.solver_evaluation import (
    build_argument_from_z3_strings,
    classify_entailment_prover9,
    classify_from_premises,
)
from src.utils import build_experiment_setting_name


def balance_trailing_parens(text: str) -> str:
    """Append missing closing parentheses to a Z3 expression."""
    open_count = text.count("(")
    close_count = text.count(")")
    if open_count > close_count:
        text += ")" * (open_count - close_count)
    return text


def balance_right_parens_per_line(text: str) -> str:
    """Append missing closing parentheses to each Prover9 input line."""
    fixed_lines = []
    for line in text.split("\n"):
        missing = line.count("(") - line.count(")")
        if missing > 0:
            suffix = ")" * missing
            line = line[:-1] + suffix + "." if line.endswith(".") else line + suffix
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def normalize_prover9_negation(text: str) -> str:
    """Normalize a common model-generated negation symbol for Prover9."""
    return re.sub(r"~", "-", text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument(
        '--normalize-solver-output',
        action='store_true',
        help='補助的にZ3/Prover9出力の括弧とProver9の否定記号を正規化する',
    )
    #parser.add_argument('--metrics_save_dir', default='./output', type=str)
    #parser.add_argument('--train_data_path', type=str)
    args = parser.parse_args()
    model_name = args.model_name
    prompt_style = args.prompt_style
    experiment_setting_name = build_experiment_setting_name(
        model_name, prompt_style, args.data_dir
    )
    file_name_json = f'output_json/output_{experiment_setting_name}.jsonl'
    output_name_csv = f'output_csv/output_{experiment_setting_name}.csv'


    # jsonl ファイルの読み込み
    ans_list = []
    gold_list = []
    category_id_list = []
    with open(file_name_json, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 30 == 0:
                print(i)
            # 1行ごとに辞書形式に変換
            record = json.loads(line)

            if "z3" in prompt_style.lower():
                if ('Answer' not in record) and ("conclusion" in record):
                    if args.normalize_solver_output:
                        record["premises"] = balance_trailing_parens(record["premises"])
                        record["conclusion"] = balance_trailing_parens(record["conclusion"])
                    try:
                        premises_conj, conclusion = build_argument_from_z3_strings(record) # convert str to FoL
                        ans = classify_from_premises(premises_conj, conclusion)
                    except:
                        ans = 'Z3ConvertError'
                    ans_dict = {"Answer": f"{ans}"}
                    record = {**record, **ans_dict}

            elif ("Prover9" in prompt_style) or ("prover9" in prompt_style):
                if args.normalize_solver_output:
                    record["prover9"] = normalize_prover9_negation(record["prover9"])
                    record["prover9"] = balance_right_parens_per_line(record["prover9"])
                try:
                    ans, info = classify_entailment_prover9(record["prover9"])
                except:
                    ans = 'Prover9ConvertError'
                ans_dict = {"Answer": f"{ans}"}
                record = {**record, **ans_dict}

            if 'Answer' not in record: #"conclusion" "Answer"ともにない場合は何かのエラー
                ans_dict = {"Answer": "NotKnownError"}
                record = {**record, **ans_dict}
            ans_list.append(record['Answer'].lower())
            gold_list.append(record['correctness of the prompt'].lower())
            if 'aac' in args.data_dir:
                category_id_list.append(record["scheme_id"])

    # DataFrameの作成
    if 'aac' in args.data_dir:
        df = pd.DataFrame({"category": category_id_list,
                           "Answer": ans_list,
                           "Gold": gold_list})
    else:
        df = pd.DataFrame({"Answer": ans_list,
                           "Gold": gold_list})
    df.to_csv(output_name_csv)
    print('end')

if __name__ == "__main__":
    main()
