from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_PROMPT_CACHE_ENV = "ENG_LLM_PROMPT_CACHE_PATH"
DEFAULT_PROMPT_CACHE_READ_ONLY_ENV = "ENG_LLM_PROMPT_CACHE_READ_ONLY"


def default_prompt_cache_path() -> Path | None:
    raw = os.getenv(DEFAULT_PROMPT_CACHE_ENV, "").strip()
    return Path(raw) if raw else None


def prompt_cache_read_only() -> bool:
    raw = os.getenv(DEFAULT_PROMPT_CACHE_READ_ONLY_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def chat_completion_with_cache(
    *,
    cache_path: str | Path | None,
    namespace: str,
    request_payload: Mapping[str, Any],
    call: Callable[[], Mapping[str, Any]],
    read_only: bool | None = None,
) -> dict[str, Any]:
    readonly_enabled = prompt_cache_read_only() if read_only is None else bool(read_only)
    if cache_path is None or not str(cache_path).strip():
        if readonly_enabled:
            raise RuntimeError("llm_prompt_cache_miss:no_cache_path")
        return dict(call())

    path = Path(cache_path)
    key_payload = {
        "namespace": namespace,
        "request": _jsonable(request_payload),
    }
    cache_key = _stable_hash(key_payload)
    cache = _load_prompt_cache(path)
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("response"), dict):
        return dict(cached["response"])
    if readonly_enabled:
        raise RuntimeError(f"llm_prompt_cache_miss:{cache_key}")

    response = dict(call())
    cache[cache_key] = {
        "request": key_payload["request"],
        "response": response,
    }
    _write_prompt_cache(path, cache)
    return response


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_prompt_cache(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _write_prompt_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_prompt_cache(path)
    merged = {**existing, **dict(cache)}
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return str(value)
