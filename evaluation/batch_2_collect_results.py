"""API Batchの完了結果を回収する。"""

from google import genai
import os
import json
import argparse

from src.batch_api import call_client
from src.solver_output import parse_premises_conclusion
from src.utils import build_experiment_setting_name, read_jsonl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    args = parser.parse_args()

    model_name = args.model_name
    prompt_style = args.prompt_style
    experiment_setting_name = build_experiment_setting_name(
        model_name, prompt_style, args.data_dir
    )
    file_name_cache = f'cache/cache_batch_{experiment_setting_name}.cache'

    #バッチ関連のファイル名
    batch_id_txt = f"batch_input/batch_id_{experiment_setting_name}.txt" # 確認用にbatchidを保存
    output_json = f'output_json/output_{experiment_setting_name}.jsonl'

    #####  ここからが実行コード  #####

    client = call_client(model_name=model_name)

    ## キャッシュ
    #file_name_cache = 'output_batch.cache'
    if os.path.exists(file_name_cache):
        # ここにはハッシュ値と生データが格納されている
        with open(file_name_cache, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = dict()


    # Batch ID を読み込み
    if not os.path.exists(batch_id_txt):
        raise FileNotFoundError("batch_id_txt が見つかりません。batch_1_submit.py を先に実行してください。")

    with open(batch_id_txt, "r") as f:
        batch_id = f.read().strip()

    print("Checking batch:", batch_id)
    batch_status = client.batches.retrieve(batch_id)

    status = batch_status.status
    print("Batch status:", status)
    #print(batch_status)

    ### エラー分析 ###
    #print(batch_status.request_counts)
    #print(batch_status.errors)        # 行番号(line)やmessageが出ることがある
    #print(batch_status.error_file_id) # ここが本丸

    #error_file_id = batch_status.error_file_id
    #txt = client.files.content(error_file_id).text  # JSONL

    ## 1行ずつエラーを表示
    #for i, line in enumerate(txt.splitlines(), 1):
    #    obj = json.loads(line)
    #    # 典型的には obj["error"] や obj["response"] 側に詳細が入ります
    #    print(f"--- error line {i} ---")
    #    print(json.dumps(obj, ensure_ascii=False, indent=2))


    # ===== 結果取得 =====
    if status == "completed":
        result_file_id = batch_status.output_file_id
        results = []
        if 'gemini' in model_name:
            genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            result_text = genai_client.files.download(file=result_file_id).decode("utf-8")
            #print(f'result_text: {result_text}')
            for line in result_text.splitlines():
                results.append(json.loads(line))
        #print(result_file_id)
        #result_stream = client.files.content(result_file_id)
        else:
            # JSONL 形式をパース
            result_stream = client.files.content(result_file_id)
            #print(f'result_stream.text: {result_stream.text.strip()}')
            for line in result_stream.text.strip().splitlines():
                results.append(json.loads(line))
        print(results[0])
        print(type(results[0]))
        print(len(results))
        # キャッシュ更新

        result_map = {}
        for num, item in enumerate(results):
            h = item["custom_id"]
            output = item["response"]["body"]["choices"][0]["message"]["content"]
            cache[h] = output
            result_map[h] = output   # ← appendではなくdictへ

        if len(result_map) != len(results):
            raise RuntimeError("custom_id collision detected (duplicate instructions). Use index-based custom_id.")

        with open(file_name_cache, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

        with open(f"batch_input/hash_list_{experiment_setting_name}.json", "r") as f:
            hash_list = json.load(f)

        # 新規取得結果と既存キャッシュの両方から、元データ順に回答を復元する。
        results_only_content = [cache[h] for h in hash_list]

        print("Batch results saved to cache.")
        #print(results_only_content)
        #出力results_only_contentを元データと合体させてjsonlで保存する
        #元データ rawdata_path = 'extracted_data_test.jsonl'
        jsonl_data = read_jsonl(args.data_dir)
        all_list = []
        error_count = 0
        if len(jsonl_data) == len(results_only_content):
            for i, (raw_dict, result_dict) in enumerate(zip(jsonl_data, results_only_content)):
                try:
                    if ("Prover9" in prompt_style) or ("prover9" in prompt_style): #prover9ではresult_dictはdictでない単なるstrなのでjsonに格納
                        result_dict = {'prover9': result_dict}
                    elif ("Z3" in prompt_style) or ("z3" in prompt_style):
                        result_dict = parse_premises_conclusion(result_dict)
                    else:
                        result_dict = json.loads(result_dict) #strをdictに
                    all_dict = {**raw_dict, **result_dict}
                    all_list.append(all_dict)
                except:
                    error_count += 1
                    print(f'error num:{i}')
                    print(result_dict)
                    print('##########')
                    all_list.append(raw_dict)
            # JSONL形式で保存
            with open(output_json, "w", encoding="utf-8") as f:
                for item in all_list:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            print('結果保存完了')
            print(f'error_count = {error_count}')
        else:
            print(f'元データの長さ {len(jsonl_data)}, 出力の長さ {len(results_only_content)}')
            print('元データと出力の長さが違う エラー')


    elif status in ["failed", "cancelled", "expired"]:
        print("Batch failed with status:", status)
    else:
        print("Batch is still running. Please check again later.")


if __name__ == "__main__":
    main()
