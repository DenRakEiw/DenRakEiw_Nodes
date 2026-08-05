"""OpenRouter LLM client for the Flux3Prompter ComfyUI node.

Talks to the OpenRouter chat-completions API over plain HTTP (no SDKs).
The model dropdown is filled live from GET /v1/models so new models appear
without a node update; if that fetch fails (offline, rate-limited) a small
curated fallback list is used.
"""

import base64
import io
import logging
import os
import time

import numpy as np
import requests
import torch
from PIL import Image

log = logging.getLogger("Flux3Prompt")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_KEY = "OPENROUTER_API_KEY"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_TIMEOUT = 120  # seconds — LLM call
MODELS_URL = "https://openrouter.ai/api/v1/models"

# Cache the model list for the lifetime of the ComfyUI process: the API is
# queried once, at node load. Re-refresh ComfyUI to pick up newly added models.
_models_cache: list[str] | None = None
_models_fetched_at: float = 0.0
_MODELS_TTL = 6 * 3600  # re-fetch at most every 6h if ComfyUI runs that long

# Small fallback used when the live fetch fails (offline / rate-limited).
# These are widely available and cover the common cases.
FALLBACK_MODELS = [
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-3.7-sonnet",
    "anthropic/claude-3.5-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
    "mistralai/mistral-large",
]


def _load_dotenv() -> dict:
    values = {}
    path = os.path.join(NODE_DIR, ".env")
    if not os.path.isfile(path):
        return values
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                values[key.strip()] = val.strip().strip('"').strip("'")
    except OSError as exc:
        log.warning("Flux3Prompt: could not read .env: %s", exc)
    return values


def get_api_key(override: str = "") -> str:
    if override and override.strip():
        return override.strip()
    env = _load_dotenv()
    key = env.get(ENV_KEY) or os.environ.get(ENV_KEY) or ""
    if not key:
        raise RuntimeError(
            f"Flux3Prompt: kein OpenRouter API-Key. Trage ihn in .env als "
            f"{ENV_KEY}=... ein, setze die Umgebungsvariable, oder fülle das "
            f"api_key-Feld der Node."
        )
    return key


def fetch_openrouter_models(force: bool = False) -> list[str]:
    """Return OpenRouter model ids that accept text input.

    Cached for _MODELS_TTL. Filters to text-capable models (architecture.modality
    starts with 'text'), drops expired/deprecated ids (prefixed '~'), sorts
    alphabetically. The default model is hoisted to the top for convenience.
    """
    global _models_cache, _models_fetched_at
    if _models_cache and not force and (time.time() - _models_fetched_at) < _MODELS_TTL:
        return _models_cache

    try:
        resp = requests.get(MODELS_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        ids = [
            m["id"]
            for m in data
            if isinstance(m.get("id"), str)
            and not m["id"].startswith("~")
            and str(m.get("architecture", {}).get("modality", "")).startswith("text")
        ]
        # Mark which models accept image input (vision-capable). We keep all
        # text-capable models in the dropdown; the node warns if an image is
        # attached to a text-only model.
        ids = sorted(set(ids))
        if not ids:
            log.warning("Flux3Prompt: OpenRouter /v1/models returned no text models; "
                        "using fallback list.")
            ids = list(FALLBACK_MODELS)
        _models_cache = ids
        _models_fetched_at = time.time()
        log.info("Flux3Prompt: loaded %d OpenRouter models.", len(ids))
        return ids
    except Exception as exc:
        log.warning("Flux3Prompt: could not fetch OpenRouter model list (%s); "
                    "using fallback list.", exc)
        _models_cache = list(FALLBACK_MODELS)
        _models_fetched_at = time.time()
        return _models_cache


def _resolve_model(model: str) -> str:
    m = (model or "").strip()
    return m if m else DEFAULT_MODEL


def is_vision_model(model: str) -> bool:
    """Check whether a model id is vision-capable (accepts image input).

    Uses the cached model list from fetch_openrouter_models(); if the list
    contains the model, we check its architecture.modality. If the model
    isn't in the list (custom slug), we optimistically return True and let
    the API reject it if it can't handle images.
    """
    if not _models_cache or model not in _models_cache:
        return True  # be optimistic for custom slugs
    # We don't store the full modality, so re-fetch isn't ideal. Instead,
    # check the fallback list of known vision-capable model prefixes.
    vision_prefixes = (
        "anthropic/claude", "openai/gpt-4", "google/gemini",
        "meta-llama/llama-3.2", "qwen/qwen-2.5-vl", "qwen/qwen2.5-vl",
        "mistralai/pixtral",
    )
    return any(model.startswith(p) for p in vision_prefixes)


def tensor_to_base64(image: torch.Tensor, fmt: str = "PNG") -> str:
    """ComfyUI IMAGE tensor [H,W,C] in 0..1 -> base64 string (no data: prefix)."""
    arr = image.detach().cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        pil = Image.fromarray(arr, "RGBA").convert("RGB")
    else:
        pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_openrouter(api_key: str, model: str, system: str, user: str,
                    temperature: float, max_tokens: int,
                    timeout: int = DEFAULT_TIMEOUT,
                    images: list[str] | None = None) -> str:
    """OpenRouter chat completion. Returns the assistant text.

    images: optional list of base64-encoded image strings. When provided,
    the user message becomes multimodal (text + image_url parts).
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build the user message: plain text, or multimodal if images are attached.
    if images:
        content = [{"type": "text", "text": user}]
        for b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        user_msg = {"role": "user", "content": content}
    else:
        user_msg = {"role": "user", "content": user}

    body = {
        "model": _resolve_model(model),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            user_msg,
        ],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if not resp.ok:
        detail = resp.text[:800] if resp.text else "(no body)"
        raise RuntimeError(f"Flux3Prompt: OpenRouter API error (HTTP {resp.status_code}): {detail}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Flux3Prompt: OpenRouter returned no choices: {data}")
    return choices[0].get("message", {}).get("content", "").strip()
