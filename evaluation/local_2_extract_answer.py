"""ローカルモデルの生出力から評価対象の回答を抽出する。"""

import argparse
import json
import re
from typing import Any, Optional, Tuple

from src.solver_output import parse_premises_conclusion
from src.utils import build_experiment_setting_name, ensure_parent_directory

def _scan_json_string_literal(s: str, start_quote_idx: int) -> Tuple[str, int]:
    """
    s[start_quote_idx] == '"' を前提に、JSONの文字列リテラルを走査して
    - 中身（エスケープ込み）を返す
    - 次に読むべき index（閉じ " の次）を返す
    """
    assert s[start_quote_idx] == '"'
    i = start_quote_idx + 1
    buf = []
    while i < len(s):
        ch = s[i]
        if ch == '\\':  # escape
            if i + 1 < len(s):
                buf.append(s[i])
                buf.append(s[i + 1])
                i += 2
                continue
            else:
                break
        if ch == '"':  # end of string literal
            return "".join(buf), i + 1
        buf.append(ch)
        i += 1
    raise ValueError("Unterminated JSON string literal in raw_response.")


def _decode_json_string_literal_content(content_with_escapes: str) -> str:
    """
    JSON文字列の中身（\" や \\n を含む）を、実際のPython文字列に戻す。
    """
    return json.loads('"' + content_with_escapes + '"')


def extract_json_after_think(txt: str) -> Optional[Any]:
    """
    txt 内の </think> の後ろにある最初の JSON（{...} or [...]）を抽出して python object にする。
    """
    m = re.search(r"</think>([\s\S]*)$", txt)
    rest = m.group(1) if m else txt

    m2 = re.search(r"([\{\[][\s\S]*)$", rest)
    if not m2:
        return None

    candidate = m2.group(1).strip()

    # 末尾を削りながら json.loads を試す（後ろに余計な文があっても拾う）
    for i in range(len(candidate), 0, -1):
        chunk = candidate[:i].strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
    return None

def extract_after_think(text: str) -> str:
    """
    Return the final response, excluding an optional thinking section.

    Conditions:
    - If '</think>' is absent, return the complete response.
    - If it appears once, return the substring after it.
    - If it appears multiple times, raise ValueError.
    """

    pattern = r"</think>"
    matches = list(re.finditer(pattern, text))

    if len(matches) == 0:
        return text
    if len(matches) > 1:
        raise ValueError("Multiple '</think>' tags found in the input text.")

    match = matches[0]
    return text[match.end():]


def extract_raw_response_txt_from_line(line: str, key: str = "raw_response") -> Optional[str]:
    """
    行文字列から raw_response の「値（JSON文字列）」だけを取り出してデコードし、txt を返す。
    行全体が壊れていても、raw_response 付近が読めれば復旧できる可能性がある。
    """
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"', line)
    if not m:
        return None

    # m.end()-1 は raw_response の値を始める " の位置
    start_quote_idx = m.end() - 1
    content_esc, _ = _scan_json_string_literal(line, start_quote_idx)
    return _decode_json_string_literal_content(content_esc)


def update_jsonl_merge_extracted_json(
    in_path: str,
    out_path: str,
    prompt_style: str,
    raw_key: str = "raw_response",
) -> None:
    rows = []

    with open(in_path, "r", encoding="utf-8") as f:
        for num_line, line in enumerate(f):
            if not line.strip():
                continue

            # まず raw_response の txt を「行文字列から」復元
            txt = extract_raw_response_txt_from_line(line, key=raw_key)
            if ("naive" in prompt_style) or ("CoT" in prompt_style):
                extracted = extract_json_after_think(txt) if isinstance(txt, str) else None
            else:
                try:
                    extracted_txt = extract_after_think(txt)
                    if ("Z3" in prompt_style) or ("z3" in prompt_style):
                        extracted = parse_premises_conclusion(extracted_txt)
                    elif ("prover9" in prompt_style) or ("Prover9" in prompt_style):
                        extracted = {"prover9": extracted_txt}
                    else:
                        extracted = None
                except (TypeError, ValueError):
                    extracted = None
            # 行全体が json.loads できるなら、それをベースに raw_response を削除して使う
            # （ここが落ちる行があるなら、最低限「raw_response以外も救いたい」ので別途対応が必要）
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # 超簡易：復旧できない行はスキップ
                continue

            row.pop(raw_key, None)
            #print(f"extracted: {extracted}")

            # extracted が dict なら merge（Answer 以外のキーもここで全部入る）
            if isinstance(extracted, dict):
                row.update(extracted)
            else:
                # dict 以外(JSONが配列など)もあり得るなら、専用キーに入れる
                print(num_line)
                print('cant convert to JSON')
                print('======')
                row["extracted_json"] = extracted

            rows.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_style', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--qwen3-only', action='store_true')
    #parser.add_argument('--metrics_save_dir', default='./output', type=str)
    #parser.add_argument('--train_data_path', type=str)
    args = parser.parse_args()
    model_name = args.model_name
    prompt_style = args.prompt_style
    experiment_setting_name = build_experiment_setting_name(
        model_name, prompt_style, args.data_dir, args.qwen3_only
    )

    file_name_json = f'output_json/raw_output_{experiment_setting_name}.jsonl'
    output_name_json = f'output_json/output_{experiment_setting_name}.jsonl'
    ensure_parent_directory(output_name_json)

    update_jsonl_merge_extracted_json(
        in_path=file_name_json,
        out_path=output_name_json,
        raw_key="raw_response",
        prompt_style = prompt_style,
    )



if __name__ == "__main__":
    main()
