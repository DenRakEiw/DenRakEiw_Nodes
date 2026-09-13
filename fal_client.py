"""HTTP client for the fal.ai Queue API (raw REST, no fal-client package).

Documented flow: POST https://queue.fal.run/<endpoint_id> returns
{request_id, status_url, response_url, cancel_url}. Poll status_url until
COMPLETED, then GET response_url for the result payload.
Spec: https://fal.ai/docs/model-endpoints/queue
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .flux3_client import _load_dotenv

log = logging.getLogger("FalAPI")

DEFAULT_BASE_URL = "https://queue.fal.run"

# nvidia/cosmos-3-super endpoints on fal.ai.
ENDPOINT_T2I = "nvidia/cosmos-3-super/text-to-image"

# fal queues can back up; ComfyUI itself never times out.
DEFAULT_TIMEOUT_MINUTES = 15
DEFAULT_TIMEOUT = DEFAULT_TIMEOUT_MINUTES * 60

# Consecutive polling failures tolerated before giving up on an already-paid job.
MAX_POLL_ERRORS = 10


def get_api_key(override: str = "") -> str:
    if override and override.strip():
        return override.strip()
    env = _load_dotenv()
    key = env.get("FAL_KEY") or os.environ.get("FAL_KEY") or ""
    if not key:
        raise RuntimeError(
            "Kein fal.ai API Key gefunden. Trage ihn in .env als "
            "FAL_KEY=... ein, setze die Umgebungsvariable FAL_KEY, "
            "oder fülle das api_key-Feld der Node."
        )
    return key


class FalClient:
    def __init__(self, api_key: str = "", base_url: str = ""):
        self.api_key = get_api_key(api_key)
        self.base_url = (base_url.rstrip("/") if base_url else DEFAULT_BASE_URL)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        })
        # Retry GETs only: replaying a POST could submit the job (and charge) twice.
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

    def submit(self, payload: dict, endpoint: str = ENDPOINT_T2I) -> dict:
        """Submit a request to the fal queue. Returns {request_id, status_url,
        response_url, ...}."""
        url = f"{self.base_url}/{endpoint}"
        size_kb = len(json.dumps(payload).encode()) / 1024
        if size_kb > 100:
            log.info("Fal: sende %.0f KB an %s", size_kb, endpoint)

        resp = self.session.post(url, json=payload, timeout=120)

        if resp.status_code in (401, 403):
            raise RuntimeError(f"Fal: API-Key ungültig oder fehlt (HTTP {resp.status_code}).")
        if not resp.ok:
            # The body says WHY the API refused (validation, payload too large, ...).
            detail = resp.text[:1000] if resp.text else "(kein Fehlertext)"
            raise RuntimeError(
                f"Fal: API lehnt den Request ab (HTTP {resp.status_code}): {detail}"
            )

        data = resp.json()
        log.info("Fal: request %s submitted (%s)", data.get("request_id"), endpoint)
        return data

    async def poll_async(self, task: dict, timeout: float = DEFAULT_TIMEOUT,
                         interval: float = 2.0) -> Any:
        """Poll status_url until COMPLETED, then fetch response_url.

        Yields the event loop so other nodes keep running (see flux3_client).
        """
        status_url = task["status_url"]
        response_url = task["response_url"]
        request_id = task["request_id"]
        started = time.monotonic()
        deadline = started + timeout
        last_status = None
        last_log = 0.0
        net_errors = 0

        while True:
            def _fetch_status():
                resp = self.session.get(status_url, timeout=60)
                try:
                    data = resp.json()
                except ValueError:
                    resp.raise_for_status()
                    raise
                # A non-JSON validation/error body on 4xx (except 429) is terminal.
                if (400 <= resp.status_code < 500 and resp.status_code != 429
                        and "status" not in data):
                    raise RuntimeError(
                        f"Fal: Polling von Request {request_id} abgelehnt (HTTP "
                        f"{resp.status_code}): "
                        f"{json.dumps(data, ensure_ascii=False)[:1000]}"
                    )
                return data

            try:
                data = await asyncio.to_thread(_fetch_status)
            except (requests.RequestException, ValueError) as exc:
                # The job is already running and paid for — a network hiccup must
                # not throw it away. Keep polling until the deadline.
                net_errors += 1
                if net_errors > MAX_POLL_ERRORS:
                    raise RuntimeError(
                        f"Fal: {net_errors} Netzwerkfehler in Folge beim Pollen von "
                        f"{request_id}. Der Job läuft serverseitig weiter — Ergebnis "
                        f"später abrufbar unter {response_url} . Letzter Fehler: {exc}"
                    ) from exc
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Fal: Timeout beim Pollen von {request_id} nach Netzwerkfehler: {exc}"
                    ) from exc
                log.warning("Fal: Netzwerkfehler beim Pollen (%d/%d), neuer Versuch in %.0fs: %s",
                            net_errors, MAX_POLL_ERRORS, interval * 2, exc)
                await asyncio.sleep(interval * 2)
                continue

            net_errors = 0
            status = data.get("status")
            waited = time.monotonic() - started

            if status != last_status:
                log.info("Fal: request %s -> %s", request_id, status)
                last_status = status
            elif waited - last_log >= 60:  # long queue: show it's alive, not hung
                log.info("Fal: request %s wartet seit %.0f min (%s)",
                         request_id, waited / 60, status)
                last_log = waited

            if status == "COMPLETED":
                def _fetch_result():
                    resp = self.session.get(response_url, timeout=120)
                    if not resp.ok:
                        detail = resp.text[:1000] if resp.text else "(kein Fehlertext)"
                        raise RuntimeError(
                            f"Fal: Ergebnis von {request_id} nicht abrufbar "
                            f"(HTTP {resp.status_code}): {detail}"
                        )
                    return resp.json()

                result = await asyncio.to_thread(_fetch_result)
                return result

            if status not in ("IN_QUEUE", "IN_PROGRESS"):
                raise RuntimeError(
                    f"Fal: Request fehlgeschlagen ({status}): "
                    f"{json.dumps(data, ensure_ascii=False)[:1000]}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"Fal: Timeout nach {timeout / 60:.0f} min (letzter Status: {status}). "
                    f"Der Job läuft serverseitig weiter — Ergebnis abrufbar unter "
                    f"{response_url} . Für lange Warteschlangen timeout_minutes erhöhen."
                )

            # Back off on long waits.
            await asyncio.sleep(min(interval * 5, interval + waited / 60))

    def download(self, url: str) -> bytes:
        """Fetch the finished asset. Retried: the result URL is signed and expires."""
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
                log.warning("Fal: Download fehlgeschlagen (Versuch %d/4), neuer Versuch in %ds: %s",
                            attempt + 1, wait, exc)
                time.sleep(wait)
        raise RuntimeError(f"Fal: Download des Ergebnisses fehlgeschlagen: {last}") from last


def format_metadata(payload: dict, result: Any, task: dict,
                    endpoint: str = ENDPOINT_T2I) -> str:
    """Full dump of everything about a run - for a Show Any node."""
    result = result if isinstance(result, dict) else {}
    lines = [
        "=== FAL.AI COSMOS 3 SUPER T2I ===",
        f"endpoint       : POST {DEFAULT_BASE_URL}/{endpoint}",
        f"request_id     : {task.get('request_id', '?')}",
        f"status_url     : {task.get('status_url', '?')}",
        f"response_url   : {task.get('response_url', '?')}",
        "",
        "--- REQUEST (an die API gesendet) ---",
    ]
    for key, value in payload.items():
        lines.append(f"{key:<15}: {value}")

    lines.append("")
    lines.append("--- RESPONSE (von der API) ---")
    for key, value in result.items():
        if key == "images" and isinstance(value, list):
            # Image entries are small dicts (url, content_type, ...) - fine to show.
            lines.append(f"{key:<15}: {json.dumps(value, indent=2, ensure_ascii=False)}")
        else:
            lines.append(f"{key:<15}: {value}")

    return "\n".join(lines)
