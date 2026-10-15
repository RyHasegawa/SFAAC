import json
import re
from pathlib import Path


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def ensure_parent_directory(file_path: str) -> None:
    """Create the parent directory for an output file if it does not exist."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


def filter_qwen3_records(records: list[dict], qwen3_only: bool) -> list[dict]:
    """Optionally retain only records selected for the Qwen3 experiment."""
    if not qwen3_only:
        return records
    return [record for record in records if record.get("Qwen3") == 1]


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
    qwen3_only: bool = False,
) -> str:
    data_name = Path(data_path).stem
    subset = "_qwen3" if qwen3_only else ""
    return f"{model_name_for_path(model_name)}_{prompt_style}_{data_name}{subset}"
