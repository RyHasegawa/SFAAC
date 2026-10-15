import json
import re
from pathlib import Path


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def model_name_for_path(model_name: str) -> str:
    """Convert a provider model ID into a filesystem-safe name."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")
    if not safe_name:
        raise ValueError(f"Model name cannot be converted to a safe path: {model_name!r}")
    return safe_name


def build_experiment_setting_name(
    model_name: str,
    prompt_style: str,
    data_path: str,
) -> str:
    data_name = Path(data_path).stem
    return f"{model_name_for_path(model_name)}_{prompt_style}_{data_name}"
