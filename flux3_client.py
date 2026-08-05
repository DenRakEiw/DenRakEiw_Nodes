"""HTTP client + media helpers for the BFL Flux 3 Video API.

Single documented endpoint: POST https://api.bfl.ai/v1/flux-3-video
Schema: discriminated union on the `mode` field (t2v / i2v / v2v / draft_enhance),
each branch strict (additionalProperties: false). Spec:
https://docs.bfl.ai/api-reference/utility/generate-a-video-with-flux-3
"""

import asyncio
import base64
import io
import json
import logging
import os
import time
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("Flux3API")

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_URL = "https://api.bfl.ai"
ENDPOINT_PATH = "v1/flux-3-video"

# One endpoint, four modes. Spec: see module docstring.
VIDEO_MODES = ["t2v", "i2v", "v2v", "draft_enhance"]

ASPECT_RATIOS = ["auto", "21:9", "2:1", "16:9", "4:3", "1:1", "3:4", "9:16"]
# Whole seconds 5..20, plus "auto".
DURATIONS = ["auto"] + [str(i) for i in range(5, 21)]
DURATION_MIN = 5
DURATION_MAX = 20
RESOLUTIONS = ["hd", "fhd"]
SAFETY_TOLERANCE_DEFAULT = 2
SAFETY_TOLERANCE_MIN = 0
SAFETY_TOLERANCE_MAX = 4
MAX_KEYFRAMES = 10

# --- Image endpoint (PRE-RELEASE, undocumented) -------------------------------
# The image API is not in the public docs yet (flux_3/flux3_image says
# "Endpoint slugs and pricing are TBA until launch"). On dev we keep the
# Flux3Image node working against the probed flux-3-large/distilled/small
# endpoints so it's ready when BFL flips the switch. Not shipped on main.
#
# These endpoints follow BFL_BASE_URL from .env (host None = use base_url).
IMAGE_ENDPOINTS = {
    "flux-3-large":     {"path": "flux-3-large",     "schema": "modes"},
    "flux-3-distilled": {"path": "flux-3-distilled", "schema": "modes"},
    "flux-3-small":     {"path": "flux-3-small",     "schema": "modes"},
}
IMAGE_MODELS = ["flux-3-large", "flux-3-distilled", "flux-3-small"]
IMAGE_ASPECT_RATIOS = [
    "1:2", "9:21", "9:16", "2:3", "3:4", "4:5", "5:7",
    "1:1",
    "7:5", "5:4", "4:3", "3:2", "16:9", "21:9", "2:1",
]
IMAGE_SIZES = ["256sq", "384sq", "512sq", "768sq", "1024sq", "2048sq", "4096sq"]
# Seed range shared with the (undocumented) image endpoints for parity; the
# documented video endpoint ignores seed entirely.
MAX_SEED = 4294967295


# Queues on the BFL side can run well past 15 minutes; ComfyUI itself never times out.
DEFAULT_TIMEOUT_MINUTES = 45
DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_MINUTES * 60

# Consecutive polling failures tolerated before giving up on an already-paid job.
MAX_POLL_ERRORS = 10

# Terminal failure states from the polling response.
STATUS_READY = "Ready"
STATUS_FAILED = {"Task not found", "Request Moderated", "Content Moderated", "Error"}


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
        log.warning("Flux3: konnte .env nicht lesen: %s", exc)
    return values


def get_api_key(override: str = "") -> str:
    if override and override.strip():
        return override.strip()
    env = _load_dotenv()
    key = env.get("BFL_API_KEY") or os.environ.get("BFL_API_KEY") or ""
    if not key:
        raise RuntimeError(
            "Kein BFL API Key gefunden. Trage ihn in Flux_3_API/.env als "
            "BFL_API_KEY=... ein, setze die Umgebungsvariable BFL_API_KEY, "
            "oder fülle das api_key-Feld der Node."
        )
    return key


def get_base_url() -> str:
    env = _load_dotenv()
    url = env.get("BFL_BASE_URL") or os.environ.get("BFL_BASE_URL") or DEFAULT_BASE_URL
    return url.rstrip("/")


# --------------------------------------------------------------------------- media

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


def batch_to_base64(images: torch.Tensor, limit: int | None = None) -> list[str]:
    out = [tensor_to_base64(img) for img in images]
    return out[:limit] if limit else out


def video_to_base64(video) -> str:
    """ComfyUI VIDEO -> base64 mp4."""
    buf = io.BytesIO()
    video.save_to(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def bytes_to_image_tensor(data: bytes) -> torch.Tensor:
    """Raw image bytes (PNG/JPEG/...) -> ComfyUI IMAGE tensor [1,H,W,3] in 0..1."""
    pil = Image.open(io.BytesIO(data))
    pil = pil.convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

# --------------------------------------------------------------------------- api

class Flux3Client:
    def __init__(self, api_key: str = "", base_url: str = ""):
        self.api_key = get_api_key(api_key)
        self.base_url = (base_url.rstrip("/") if base_url else get_base_url())
        self.session = requests.Session()
        self.session.headers.update(
            {"x-key": self.api_key, "accept": "application/json"}
        )
        # requests retries nothing by default, so a single dropped keep-alive
        # connection would kill a job that already cost credits. Retry GETs only:
        # replaying a POST could submit the job (and charge) twice.
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def submit(self, payload: dict, image_model: str = "") -> dict:
        """Submit a generation task.

        image_model: when set to one of IMAGE_MODELS, POSTs to that (pre-release,
        undocumented) image endpoint instead of the documented video endpoint.
        """
        if image_model:
            ep = IMAGE_ENDPOINTS.get(image_model)
            if not ep:
                raise ValueError(
                    f"Flux3: unbekanntes image model '{image_model}'. "
                    f"Erlaubt: {', '.join(IMAGE_MODELS)}."
                )
            path = f"v1/{ep['path']}"
            url = f"{self.base_url}/{path}"
            what = image_model
        else:
            path = ENDPOINT_PATH
            url = f"{self.base_url}/{ENDPOINT_PATH}"
            what = ENDPOINT_PATH
        size_mb = len(json.dumps(payload).encode()) / (1024 * 1024)
        if size_mb > 1:
            log.info("Flux3: sende %.1f MB an %s (mode=%s)",
                     size_mb, what, payload.get("mode"))

        resp = self.session.post(url, json=payload, timeout=300)

        if resp.status_code in (401, 403):
            raise RuntimeError(f"Flux3: API-Key ungültig oder fehlt (HTTP {resp.status_code}).")
        if not resp.ok:
            # raise_for_status() would swallow the body - and the body is the whole point:
            # it says WHY the API refused (bad base64, payload too large, moderation, ...).
            detail = resp.text[:1000] if resp.text else "(kein Fehlertext)"
            hint = ""
            if resp.status_code in (400, 413) and size_mb > 5:
                hint = (f" Der Request war {size_mb:.1f} MB groß — vermutlich ist das "
                        f"start_video/keyframes/input_image zu groß.")
            raise RuntimeError(
                f"Flux3: API lehnt den Request ab (HTTP {resp.status_code}) "
                f"für mode={payload.get('mode')}: {detail}{hint}"
            )

        data = resp.json()
        cost = data.get("cost")
        log.info("Flux3: task %s submitted (cost: %s credits)", data.get("id"), cost)
        return data


    def poll(self, task: dict, timeout: float = 900.0, interval: float = 2.0) -> Any:
        polling_url = task["polling_url"]
        task_id = task["id"]
        deadline = time.monotonic() + timeout
        last_status = None

        while True:
            resp = self.session.get(polling_url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status != last_status:
                log.info("Flux3: task %s -> %s", task_id, status)
                last_status = status

            if status == STATUS_READY:
                result = data.get("result")
                if not result:
                    raise RuntimeError(f"Flux3: Task fertig, aber ohne Ergebnis: {data}")
                return result
            if status in STATUS_FAILED:
                raise RuntimeError(
                    f"Flux3: Task fehlgeschlagen ({status}): {data.get('details')}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Flux3: Timeout nach {timeout:.0f}s (letzter Status: {status})"
                )
            time.sleep(interval)

    async def poll_async(self, task: dict, timeout: float = DEFAULT_TIMEOUT,
                         interval: float = 2.0) -> Any:
        """Same as poll(), but yields the event loop so other nodes keep running.

        ComfyUI runs coroutine FUNCTIONs concurrently: a node that awaits is parked as
        PENDING while independent nodes execute. A blocking time.sleep() here would
        stall the whole graph, so every wait and every HTTP call has to yield.
        """
        polling_url = task["polling_url"]
        task_id = task["id"]
        started = time.monotonic()
        deadline = started + timeout
        last_status = None
        last_log = 0.0
        net_errors = 0

        while True:
            def _fetch():
                resp = self.session.get(polling_url, timeout=60)
                resp.raise_for_status()
                return resp.json()

            try:
                data = await asyncio.to_thread(_fetch)
            except (requests.RequestException, ValueError) as exc:
                # The job is already running and paid for — a network hiccup must not
                # throw it away. Keep polling until the deadline.
                net_errors += 1
                if net_errors > MAX_POLL_ERRORS:
                    raise RuntimeError(
                        f"Flux3: {net_errors} Netzwerkfehler in Folge beim Pollen von "
                        f"{task_id}. Der Job läuft serverseitig weiter — Ergebnis später "
                        f"abrufbar unter {polling_url} . Letzter Fehler: {exc}"
                    ) from exc
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Flux3: Timeout beim Pollen von {task_id} nach Netzwerkfehler: {exc}"
                    ) from exc
                log.warning("Flux3: Netzwerkfehler beim Pollen (%d/%d), neuer Versuch in %.0fs: %s",
                            net_errors, MAX_POLL_ERRORS, interval * 2, exc)
                await asyncio.sleep(interval * 2)
                continue

            net_errors = 0
            status = data.get("status")
            waited = time.monotonic() - started

            if status != last_status:
                log.info("Flux3: task %s -> %s", task_id, status)
                last_status = status
            elif waited - last_log >= 60:  # long queue: show it's alive, not hung
                progress = data.get("progress")
                log.info("Flux3: task %s wartet seit %.0f min (%s%s)",
                         task_id, waited / 60, status,
                         f", {progress:.0%}" if isinstance(progress, (int, float)) else "")
                last_log = waited

            if status == STATUS_READY:
                result = data.get("result")
                if not result:
                    raise RuntimeError(f"Flux3: Task fertig, aber ohne Ergebnis: {data}")
                return result
            if status in STATUS_FAILED:
                raise RuntimeError(
                    f"Flux3: Task fehlgeschlagen ({status}): {data.get('details')}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Flux3: Timeout nach {timeout / 60:.0f} min (letzter Status: {status}). "
                    f"Der Job läuft serverseitig weiter — Ergebnis abrufbar unter "
                    f"{polling_url} . Für lange Warteschlangen timeout_minutes erhöhen."
                )

            # Back off on long waits: no point hammering the polling_url every 2s for 20 minutes.
            await asyncio.sleep(min(interval * 5, interval + waited / 60))

    def download(self, url: str) -> bytes:
        """Fetch the finished asset. Retried: the result URL is signed and expires,
        so a dropped connection here would lose the video for good."""
        last: Exception | None = None
        for attempt in range(4):
            try:
                # Plain requests.get() - the signed delivery URL must not carry our api key.
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as exc:
                last = exc
                wait = 2 ** attempt
                log.warning("Flux3: Download fehlgeschlagen (Versuch %d/4), neuer Versuch in %ds: %s",
                            attempt + 1, wait, exc)
                time.sleep(wait)
        raise RuntimeError(f"Flux3: Download des Ergebnisses fehlgeschlagen: {last}") from last


# Payload keys whose values are base64 blobs - never dump those into the debug output.
BLOB_KEYS = ("keyframes", "start_video", "draft_cache", "input_image")


def _describe_blob(value: Any) -> str:
    """Render a base64 input as a size, not as megabytes of gibberish."""
    if isinstance(value, list):
        return "[" + ", ".join(_describe_blob(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {_describe_blob(v)}" for k, v in value.items())
        return "{" + inner + "}"
    # [seconds, base64] pair: a 2-element list reaches _describe_blob via the list branch above,
    # but a bare tuple would land here.
    if isinstance(value, tuple):
        return "[" + ", ".join(_describe_blob(v) for v in value) + "]"
    if isinstance(value, str):
        if value.startswith("http"):
            return value
        kb = len(value) * 3 / 4 / 1024  # base64 -> bytes
        return f"<base64, {kb:.0f} KB>"
    return repr(value)


def format_metadata(payload: dict, result: Any, task: dict,
                    image_model: str = "") -> str:
    """Full, untruncated dump of everything about a run - for a Show Any node."""
    result = result if isinstance(result, dict) else {}
    if image_model:
        path = f"v1/{IMAGE_ENDPOINTS.get(image_model, {}).get('path', image_model)}"
        header = "=== FLUX 3 IMAGE (pre-release) ==="
    else:
        path = ENDPOINT_PATH
        header = "=== FLUX 3 VIDEO ==="
    lines = [
        header,
        f"endpoint       : POST {get_base_url()}/{path}",
        f"task_id        : {task.get('id', '?')}",
        f"polling_url    : {task.get('polling_url', '?')}",
    ]
    for key in ("cost", "input_mp", "output_mp"):
        if task.get(key) is not None:
            lines.append(f"{key:<15}: {task[key]}")

    lines.append("")
    lines.append("--- REQUEST (an die API gesendet) ---")
    for key, value in payload.items():
        shown = _describe_blob(value) if key in BLOB_KEYS else value
        lines.append(f"{key:<15}: {shown}")

    lines.append("")
    lines.append("--- RESPONSE (von der API) ---")
    for key, value in result.items():
        lines.append(f"{key:<15}: {value}")

    return "\n".join(lines)


def extract_url(result: Any) -> str:
    """The result payload puts the asset under `sample` as a signed .mp4 URL."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("sample", "video", "url", "output"):
            val = result.get(key)
            if isinstance(val, str) and val.startswith("http"):
                return val
        for val in result.values():
            if isinstance(val, str) and val.startswith("http"):
                return val
    raise RuntimeError(f"Flux3: keine Ergebnis-URL gefunden in: {result}")
