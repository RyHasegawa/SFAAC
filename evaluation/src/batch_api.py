from openai import OpenAI
from google import genai                   # Google 公式 Gemini クライアント
from google.genai import types             # ファイルアップロード設定
import os
import hashlib
import json

def call_client(model_name):
    if 'gpt' in model_name:
        client = OpenAI(
            organization = os.getenv("OPENAI_ORGANIZATION_KEY"),
            api_key = os.getenv("OPENAI_API_KEY"),
        )
    elif 'gemini' in model_name:
        client = OpenAI(
            organization = os.getenv("OPENAI_ORGANIZATION_KEY"),
            api_key = os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        raise ValueError(f"Unsupported API model: {model_name}")
    return client

#ハッシュ系の関数
def create_hash(prompt):
    return hashlib.md5(prompt.encode()).hexdigest() # instruction単位で保存

def load_hash_json(file_name):
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)

# batch

def add_to_batch_if_needed(instruction, batch_requests, cache, model_name, system_text, user_text, assistant_text, response_format):
    """キャッシュにない場合、Batch API 用リクエストを追加"""
    h = create_hash(instruction)
    if h not in cache:
        if 'gpt-4' in model_name:
            batch_requests.append({
                "custom_id": h,  # 後で対応付ける用
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": instruction}
                ],
                    "response_format": response_format,
                    "seed": 1,
                    "n": 1,
                    "temperature": 0.0, #0.5 #gpt-4
                    "max_tokens": 2500 #gpt-4
                }
            })

        elif 'gpt' in model_name:
            batch_requests.append({
                "custom_id": h,  # 後で対応付ける用
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": instruction}
                ],
                    "response_format": response_format,
                    "seed": 1,
                    "n": 1,
                    "max_completion_tokens": 7000,
                    'reasoning_effort': 'medium',
                }
            })

        elif 'gemini' in model_name:
            batch_requests.append({
                "custom_id": h,  # 後で対応付ける用
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
                {"role": "user", "content": instruction}
                ],
                    "response_format": response_format,
                    "max_completion_tokens": 7000, #gpt-5
                    'reasoning_effort': 'medium', #gpt-5
                }
            })
    return h

def execute_batch(batch_requests, batch_input_file, batch_id_txt, model_name):
    # ===== Batch API の投入 =====
    if batch_requests:
        #batch_input_file = "batch_for_openAIapi.jsonl"
        with open(batch_input_file, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")


        if 'gemini' in model_name:
            # geminiの場合、ファイルアップロードだけはOpenAI 互換 APIではなく
            # Google GeminiAPI（native）を使う必要がある
            client = call_client(model_name)
            google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            uploaded_file = google_client.files.upload(
                file=open(batch_input_file, "rb"),
                config=types.UploadFileConfig(
                    display_name="batch",
                    mime_type="jsonl"
                ),
            )
            input_file_id=uploaded_file.name
        else:
            # ファイルをアップロード
            client = call_client(model_name)
            uploaded_file = client.files.create(file=open(batch_input_file, "rb"), purpose="batch")
            input_file_id=uploaded_file.id

        # Batch を作成
        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        #print("RAW batch:", batch)
        #print("batch.id  :", repr(getattr(batch, "id", None)))
        #print("batch.name:", repr(getattr(batch, "name", None)))

        # ここでいったん直接呼んでみる
        #check = client.batches.retrieve(batch.id)
        #print("status:", check.status)

        #idを保存
        print("Batch job submitted:", batch.id)
        with open(batch_id_txt, "w") as f:
            f.write(batch.id)

        print('end')

    else:
        print("すべてキャッシュ済みです。")
