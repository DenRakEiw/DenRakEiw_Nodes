"""ComfyUI nodes for the BFL Flux 3 API.

Flux3Video — documented endpoint POST /v1/flux-3-video, four modes
(t2v / i2v / v2v / draft_enhance).
Spec: https://docs.bfl.ai/api-reference/utility/generate-a-video-with-flux-3

Flux3VideoUpscale — documented endpoint POST /v1/flux-tools/video-upscale-v1,
super-resolution for short clips (≤20s, ≤50MB) with a precise and a creative
mode. Output capped at ~14.4 MP per frame; source audio preserved.
Spec: https://docs.bfl.ai/api-reference/utility/video-upscale-v1

Flux3Prompter — LLM-powered prompt generator (OpenRouter) that turns a vague
idea into a structured FLUX 3 prompt, guided by a prompting skill.
"""

import asyncio
import glob
import io
import logging
import os
from typing import Any

import torch

from .flux3_client import (
    ASPECT_RATIOS,
    DEFAULT_TIMEOUT_MINUTES,
    DURATION_MAX,
    DURATION_MIN,
    DURATIONS,
    MAX_KEYFRAMES,
    RESOLUTIONS,
    SAFETY_TOLERANCE_DEFAULT,
    SAFETY_TOLERANCE_MAX,
    SAFETY_TOLERANCE_MIN,
    UPSCALE_CREATIVITY_MODES,
    UPSCALE_ENDPOINT_PATH,
    UPSCALE_FACTOR_MAX,
    UPSCALE_FACTOR_MIN,
    UPSCALE_TARGETS,
    UPSCALE_TARGET_SHORT_SIDE,
    UPSCALE_VIDEO_MAX_MB,
    UPSCALE_VIDEO_MAX_SECONDS,
    VIDEO_MODES,
    Flux3Client,
    batch_to_base64,
    extract_url,
    format_metadata,
    tensor_to_base64,
    video_to_base64,
)
from .llm_client import (
    DEFAULT_MODEL as OR_DEFAULT_MODEL,
    call_openrouter,
    fetch_openrouter_models,
    get_api_key as get_openrouter_api_key,
    is_vision_model,
    tensor_to_base64 as image_to_base64,
)

log = logging.getLogger("Flux3API")

from comfy_api.input_impl import VideoFromFile

CATEGORY = "Flux3 API"


async def _run(client: Flux3Client, payload: dict,
               timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES) -> tuple[dict, dict, bytes]:
    """Submit, poll and download without blocking the executor.

    Every step is either awaited or pushed onto a thread, so ComfyUI can run other
    Flux nodes in the same graph concurrently instead of one after another.
    """
    task = await asyncio.to_thread(client.submit, payload)
    result = await client.poll_async(task, timeout=timeout_minutes * 60)
    data = await asyncio.to_thread(client.download, extract_url(result))
    return task, result, data



def _parse_keyframe_times(text: str) -> list[float]:
    """'0, 4.5, 10' -> [0.0, 4.5, 10.0]. Accepts commas or semicolons."""
    try:
        return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(
            f"Flux3: keyframe_times muss eine Liste von Sekunden sein (z.B. '0, 4.5, 10'), "
            f"bekommen: {text!r}"
        ) from exc


async def _build_keyframes(images: torch.Tensor | None, end_image: torch.Tensor | None,
                           keyframe_times: str, duration_value: str) -> Any:
    """Build the `keyframes` payload for i2v per the documented schema.

    - 1 image, no times: opening frame           -> str
    - 2 images (or images + end_image): start+end -> [str, str]
    - 3+ images, no times: evenly spread          -> [str, ...]  (duration muss gesetzt sein)
    - images + keyframe_times: storyboard         -> [[seconds, str], ...]
    """
    if images is None:
        raise ValueError("Flux3: i2v braucht einen images-Input.")

    # images + end_image ist die bequeme Art, Start+Ende zu stecken.
    if end_image is not None:
        imgs = torch.cat([images[:1], end_image[:1]])
    else:
        imgs = images

    n = len(imgs)
    if n < 1:
        raise ValueError("Flux3: i2v braucht mindestens ein Bild am images-Input.")
    if n > MAX_KEYFRAMES:
        raise ValueError(
            f"Flux3: i2v nimmt höchstens {MAX_KEYFRAMES} Keyframes, der Batch hat {n}."
        )

    times_text = keyframe_times.strip()
    if times_text:
        times = _parse_keyframe_times(times_text)
        if len(times) != n:
            raise ValueError(
                f"Flux3: keyframe_times braucht genauso viele Werte wie Bilder — "
                f"{n} Bild(er), aber {len(times)} Zeit(en)."
            )
        if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
            raise ValueError(
                f"Flux3: keyframe_times müssen aufsteigend sein, bekommen: {times}"
            )
        if any(t < 0 for t in times):
            raise ValueError(f"Flux3: keyframe_times dürfen nicht negativ sein: {times}")
        encoded = await asyncio.to_thread(batch_to_base64, imgs)
        return [[t, b64] for t, b64 in zip(times, encoded)]

    if n >= 3 and duration_value == "auto":
        # Spec: "3 or more need a set duration."
        raise ValueError(
            "Flux3: 3+ Keyframes ohne keyframe_times brauchen eine feste duration "
            "(5–20). Stelle duration auf einen Wert, oder gib keyframe_times an."
        )

    encoded = await asyncio.to_thread(batch_to_base64, imgs)
    if n == 1:
        return encoded[0]
    return encoded  # list[str]: 2 -> start+end, 3+ -> evenly spread


class Flux3Video:
    """FLUX 3 Video — t2v / i2v / v2v / draft_enhance, mit synchronem Audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (VIDEO_MODES, {
                    "default": "t2v",
                    "tooltip": "t2v: Text→Video · "
                               "i2v: Bild(er)→Video (keyframes) · "
                               "v2v: Clip fortsetzen (start_video) · "
                               "draft_enhance: einen vorherigen draft final rendern."}),
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Pflicht für t2v/i2v/v2v. Bei draft_enhance nicht "
                               "erlaubt — der draft_cache pins alles."}),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "auto",
                    "tooltip": "'auto' läßt die API anhand von Prompt und Referenzen wählen."}),
                "duration": (DURATIONS, {
                    "default": "auto",
                    "tooltip": "Ganze Sekunden 5–20, oder 'auto'. 3+ Keyframes ohne "
                               "keyframe_times brauchen eine feste Länge."}),
                "resolution": (RESOLUTIONS, {
                    "default": "hd",
                    "tooltip": "hd = Default · fhd = höhere Auflösung (Video-Upscaler)."}),
                "generate_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Default an. Aus = stummer Clip."}),
                "safety_tolerance": ("INT", {
                    "default": SAFETY_TOLERANCE_DEFAULT,
                    "min": SAFETY_TOLERANCE_MIN, "max": SAFETY_TOLERANCE_MAX,
                    "tooltip": "0 (strengste) bis 4. Default 2. Mit Conditioning-Media "
                               "maximal 2."}),
                "draft": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Schnelle hd-Vorschau. Das Ergebnis enthält einen "
                               "draft_cache, den man mit mode=draft_enhance final "
                               "rendern kann."}),
                "version": ("STRING", {
                    "default": "latest",
                    "tooltip": "Aktuell nur 'latest' freigegeben."}),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Nur i2v: 1 Bild (Startframe) · 2 Bilder (Start+Ende) · "
                               "3–10 Bilder (gleichmäßig verteilt, braucht duration) · "
                               "oder zusammen mit keyframe_times (Storyboard)."}),
                "end_image": ("IMAGE", {
                    "tooltip": "Nur i2v: wenn gesetzt, gelten images=Start und "
                               "end_image=Ende (genau 2 Keyframes)."}),
                "keyframe_times": ("STRING", {
                    "default": "",
                    "tooltip": "Nur i2v: Sekunden je Bild, kommagetrennt (z.B. '0, 4.5, 10'). "
                               "Anzahl muss mit images übereinstimmen, Werte aufsteigend. "
                               "Bilder werden zu Frames an diesen Sekunden."}),
                "video": ("VIDEO", {
                    "tooltip": "Nur v2v: der Clip, der fortgesetzt wird (mp4)."}),
                "draft_cache": ("STRING", {
                    "default": "",
                    "tooltip": "Nur draft_enhance: base64-Bundle oder URL aus dem "
                               "draft-Output eines vorherigen Laufs. Pflicht für diesen Modus."}),
                "timeout_minutes": ("INT", {
                    "default": DEFAULT_TIMEOUT_MINUTES, "min": 1, "max": 240,
                    "tooltip": "Wie lange die Node auf das Ergebnis wartet. Bei voller "
                               "BFL-Warteschlange dauert ein Job schnell 20+ Minuten. "
                               "Läuft die Zeit ab, bricht nur die Node ab — der Job läuft "
                               "serverseitig weiter (und kostet trotzdem Credits)."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Leer = aus .env"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "metadata")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    async def generate(self, mode, prompt, aspect_ratio, duration, resolution,
                       generate_audio, safety_tolerance, draft, version,
                       images=None, end_image=None, keyframe_times="", video=None,
                       draft_cache="",
                       timeout_minutes=DEFAULT_TIMEOUT_MINUTES, api_key=""):
        if mode not in VIDEO_MODES:
            raise ValueError(
                f"Flux3: Modus '{mode}' nicht unterstützt. Erlaubt: {', '.join(VIDEO_MODES)}.")

        # --- draft_enhance: only mode, draft_cache, safety_tolerance are accepted.
        if mode == "draft_enhance":
            cache = draft_cache.strip()
            if not cache:
                raise ValueError(
                    "Flux3: draft_enhance braucht draft_cache (base64-Bundle oder URL "
                    "aus einem vorherigen draft-Lauf)."
                )
            payload = {
                "mode": "draft_enhance",
                "draft_cache": cache,
                "safety_tolerance": int(safety_tolerance),
            }
            # The bundle pins everything else; the API rejects any further field
            # (additionalProperties: false). Warn about inputs that would be silently dropped.
            ignored = [n for n, used in (
                ("prompt", bool(prompt.strip())),
                ("images", images is not None),
                ("end_image", end_image is not None),
                ("keyframe_times", bool(keyframe_times.strip())),
                ("video", video is not None),
                ("aspect_ratio", aspect_ratio != "auto"),
                ("duration", duration != "auto"),
                ("resolution", resolution != "hd"),
                ("generate_audio", not generate_audio),
                ("draft", draft),
                ("version", version.strip() not in ("", "latest")),
            ) if used]
            if ignored:
                log.warning("Flux3: draft_enhance akzeptiert nur draft_cache + safety_tolerance. "
                            "Diese Inputs werden ignoriert: %s", ", ".join(ignored))
        else:
            if not prompt.strip():
                raise ValueError("Flux3: prompt darf nicht leer sein.")

            payload: dict = {"mode": mode, "prompt": prompt.strip()}

            if aspect_ratio not in ASPECT_RATIOS:
                raise ValueError(
                    f"Flux3: aspect_ratio '{aspect_ratio}' nicht unterstützt. "
                    f"Erlaubt: {', '.join(ASPECT_RATIOS)}."
                )
            payload["aspect_ratio"] = aspect_ratio

            # duration: "auto" oder int 5..20
            if duration not in DURATIONS:
                raise ValueError(
                    f"Flux3: duration '{duration}' nicht unterstützt. "
                    f"Erlaubt: auto oder ganze Sekunden 5–20."
                )
            payload["duration"] = duration if duration == "auto" else int(duration)

            if resolution not in RESOLUTIONS:
                raise ValueError(
                    f"Flux3: resolution '{resolution}' nicht unterstützt. "
                    f"Erlaubt: {', '.join(RESOLUTIONS)}."
                )
            payload["resolution"] = resolution

            payload["generate_audio"] = bool(generate_audio)

            if not (SAFETY_TOLERANCE_MIN <= safety_tolerance <= SAFETY_TOLERANCE_MAX):
                raise ValueError(
                    f"Flux3: safety_tolerance muss {SAFETY_TOLERANCE_MIN}..{SAFETY_TOLERANCE_MAX} "
                    f"sein, bekommen: {safety_tolerance}."
                )
            payload["safety_tolerance"] = int(safety_tolerance)

            if draft:
                payload["draft"] = True

            v = version.strip()
            if v and v != "latest":
                # Spec enum currently only allows "latest"; keep the field ready for dated tags.
                payload["version"] = v

            # --- mode-specific conditioning input ---
            if mode == "i2v":
                payload["keyframes"] = await _build_keyframes(
                    images, end_image, keyframe_times, duration)

            elif mode == "v2v":
                if video is None:
                    raise ValueError("Flux3: v2v braucht einen video-Input (start_video).")
                payload["start_video"] = await asyncio.to_thread(video_to_base64, video)

            # t2v: no extra conditioning field.

        client = Flux3Client(api_key)
        task, result, data = await _run(client, payload, timeout_minutes)
        return (VideoFromFile(io.BytesIO(data)),
                format_metadata(payload, result, task))


# ===========================================================================
# Flux3VideoUpscale — super-resolution for short clips (video-upscale-v1)
# ===========================================================================

class Flux3VideoUpscale:
    """FLUX Video Upscale — super-resolution for short clips.

    POST /v1/flux-tools/video-upscale-v1. Source clip max 20 s / 50 MB; output
    capped at ~14.4 MP per frame (very large sources get upscaled by less than
    the requested factor). The source audio track is preserved.

    Two modes via `creativity`:
      - precise (0): preserves the source exactly and sharpens it. Use when
        identity matters (faces, products, brand assets, real people).
      - creative (1, default): restores/invents fine detail more aggressively.
        Good for generated footage, textures, crowds, scenery. Does not strictly
        preserve identity — faces/products can drift.

    Spec: https://docs.bfl.ai/api-reference/utility/video-upscale-v1
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": "Der Clip, der upgescalet wird (mp4). Max "
                               f"{UPSCALE_VIDEO_MAX_SECONDS} s und "
                               f"{UPSCALE_VIDEO_MAX_MB} MB. Alternativ "
                               "input_video_url verwenden."}),
                "target_resolution": (UPSCALE_TARGETS, {
                    "default": "4K",
                    "tooltip": "Ziel-Auflösung (Short Side). Die Node misst die "
                               "Source-Dimension und berechnet den upscale_factor "
                               "daraus (Faktor = target_short / min(w, h), "
                               "auf 1.5–3.0 begrenzt). Erhält das Seitenverhältnis "
                               "der Quelle. 1080p ≈ 1920×1080, 2K ≈ 2560×1440, "
                               "4K ≈ 3840×2160 (jeweils 16:9)."}),
                "creativity": (UPSCALE_CREATIVITY_MODES, {
                    "default": "creative",
                    "tooltip": "precise (0) = identitätserhaltend, scharf — für "
                               "Gesichter/Produkte/echte Menschen. "
                               "creative (1, default) = erfindet Detail, gut für "
                               "generierte Footage/Landschaften/Texturen; "
                               "Gesichter/Produkte können driften."}),
            },
            "optional": {
                "input_video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Alternative: HTTP(S)-URL zur Quelle statt "
                               "video-Input. Hat Vorrang vor dem video-Input; "
                               "erspart das Base64-Encoding eines großen Clips. "
                               "Die Node streamt nur den moov-Atom der URL, um "
                               "die Source-Dimension zu messen — kein Full-Download."}),
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional Beschreibung des Clips, lenkt das "
                               "enhanced Detail (vor allem im creative-Modus). "
                               "Leer = neutraler Upscale."}),
                "safety_tolerance": ("INT", {
                    "default": SAFETY_TOLERANCE_DEFAULT,
                    "min": SAFETY_TOLERANCE_MIN, "max": SAFETY_TOLERANCE_MAX,
                    "tooltip": "0 (strengste) bis 4, Default 2. Moderation für "
                               "Prompt und ausgelieferte Frames."}),
                "timeout_minutes": ("INT", {
                    "default": DEFAULT_TIMEOUT_MINUTES, "min": 1, "max": 240,
                    "tooltip": "Wie lange die Node auf das Ergebnis wartet. "
                               "Upscale kann ähnlich lange wie eine Video-"
                               "Generierung brauchen."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Leer = aus .env"}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "metadata")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @staticmethod
    def _probe_url_dimensions(url: str) -> tuple[int, int]:
        """Stream-probe a remote clip's (width, height) without downloading it.

        PyAV only pulls the moov atom (kB, not MB) when you open a remote
        source and read the first video stream's codec context. We don't
        decode a single frame.
        """
        import av
        with av.open(url, mode="r") as container:
            for stream in container.streams:
                if stream.type == "video":
                    return int(stream.width), int(stream.height)
        raise ValueError(f"Flux3: kein Video-Stream in {url}")

    async def generate(self, video, target_resolution, creativity,
                       input_video_url="", prompt="",
                       safety_tolerance=SAFETY_TOLERANCE_DEFAULT,
                       timeout_minutes=DEFAULT_TIMEOUT_MINUTES, api_key=""):
        if target_resolution not in UPSCALE_TARGET_SHORT_SIDE:
            raise ValueError(
                f"Flux3: target_resolution '{target_resolution}' nicht unterstützt. "
                f"Erlaubt: {', '.join(UPSCALE_TARGETS)}."
            )
        target_short = UPSCALE_TARGET_SHORT_SIDE[target_resolution]

        # --- Source dimensions: URL takes priority; otherwise the VIDEO input.
        url = (input_video_url or "").strip()
        if url:
            if not url.startswith("http"):
                raise ValueError(
                    "Flux3: input_video_url muss eine HTTP(S)-URL sein."
                )
            src_w, src_h = await asyncio.to_thread(
                self._probe_url_dimensions, url)
            input_video = url
        else:
            if video is None:
                raise ValueError(
                    "Flux3: upscale braucht entweder den video-Input oder "
                    "input_video_url."
                )
            src_w, src_h = video.get_dimensions()
            input_video = await asyncio.to_thread(video_to_base64, video)

        # --- Compute the upscale_factor from the source short side.
        source_short = min(src_w, src_h)
        factor = target_short / source_short
        if factor < UPSCALE_FACTOR_MIN:
            raise ValueError(
                f"Flux3: Source ist {src_w}x{src_h} (short side {source_short}px) "
                f"und damit bereits ≥ {target_resolution} (short side "
                f"{target_short}px). Hochskalieren ergäbe Faktor "
                f"{factor:.2f} < min {UPSCALE_FACTOR_MIN} — nichts zu upscaling. "
                f"Wähle ein höheres target_resolution (z.B. 4K) oder eine "
                f"kleinere Quelle."
            )
        if factor > UPSCALE_FACTOR_MAX:
            log.warning(
                "Flux3: %s aus Source %dx%d (short %d) nicht erreichbar "
                "(Faktor wäre %.2f, max %.1f). Upcaling mit %.1fx — effektive "
                "Short Side ~%dpx.",
                target_resolution, src_w, src_h, source_short,
                factor, UPSCALE_FACTOR_MAX, UPSCALE_FACTOR_MAX,
                int(source_short * UPSCALE_FACTOR_MAX),
            )
            factor = UPSCALE_FACTOR_MAX

        if creativity not in UPSCALE_CREATIVITY_MODES:
            raise ValueError(
                f"Flux3: creativity '{creativity}' nicht unterstützt. "
                f"Erlaubt: {', '.join(UPSCALE_CREATIVITY_MODES)}."
            )
        creativity_val = 1 if creativity == "creative" else 0

        if not (SAFETY_TOLERANCE_MIN <= safety_tolerance <= SAFETY_TOLERANCE_MAX):
            raise ValueError(
                f"Flux3: safety_tolerance muss {SAFETY_TOLERANCE_MIN}..{SAFETY_TOLERANCE_MAX} "
                f"sein, bekommen: {safety_tolerance}."
            )

        payload: dict = {
            "input_video": input_video,
            "upscale_factor": float(factor),
            "creativity": int(creativity_val),
            "safety_tolerance": int(safety_tolerance),
        }
        if prompt.strip():
            payload["prompt"] = prompt.strip()

        client = Flux3Client(api_key)
        task = await asyncio.to_thread(client.submit_upscale, payload)
        result = await client.poll_async(task, timeout=timeout_minutes * 60)
        data = await asyncio.to_thread(client.download, extract_url(result))

        meta = format_metadata(
            payload, result, task,
            endpoint_path=UPSCALE_ENDPOINT_PATH,
            header_override="=== FLUX 3 VIDEO UPSCALE ===",
        )
        meta += (
            f"\nsource_resolution : {src_w}x{src_h}\n"
            f"target_resolution : {target_resolution} (short side {target_short}px)\n"
            f"computed_factor   : {factor:.2f}"
        )
        return (VideoFromFile(io.BytesIO(data)), meta)


# ===========================================================================
# Flux3Prompter — LLM-powered prompt generator (OpenRouter)
# ===========================================================================

_SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")

_PROMPTER_SYSTEM_FRAME = (
    "You are helping a ComfyUI user craft a FLUX 3 video generation prompt.\n"
    "Follow the prompting skill below exactly — its structure, section format, "
    "tag system, timing rules, camera vocabulary, and output standard.\n"
    "Turn the user's idea into a complete, ready-to-paste FLUX 3 prompt.\n\n"
    "IMPORTANT OVERRIDE OF THE SKILL'S OUTPUT STANDARD:\n"
    "- Output ONLY the single main prompt. No style variants, no alternative versions.\n"
    "- No pacing recommendations, no dialogue word counts, no delta summaries,\n"
    "  no character counts, no metadata of any kind.\n"
    "- No preamble, no explanation, no commentary before or after the prompt.\n"
    "- The entire output must be the prompt itself and nothing else — ready to\n"
    "  paste directly into a FLUX 3 generation node.\n"
)


def _list_skills() -> list[str]:
    names = ["none"]
    if os.path.isdir(_SKILLS_DIR):
        for path in sorted(glob.glob(os.path.join(_SKILLS_DIR, "*.md"))):
            names.append(os.path.splitext(os.path.basename(path))[0])
    return names


def _load_skill(name: str) -> str:
    if not name or name == "none":
        return ""
    path = os.path.join(_SKILLS_DIR, f"{name}.md")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Flux3Prompt: skill '{name}' nicht gefunden in {_SKILLS_DIR}.")
    with open(path, "r", encoding="utf-8-sig") as fh:
        return fh.read().strip()


def _model_dropdown() -> list[str]:
    models = fetch_openrouter_models()
    if OR_DEFAULT_MODEL in models:
        models = [OR_DEFAULT_MODEL] + [m for m in models if m != OR_DEFAULT_MODEL]
    return models


class Flux3Prompter:
    """LLM-powered FLUX 3 prompt generator (OpenRouter)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "idea": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Describe your video idea in plain words. The LLM "
                               "turns it into a structured FLUX 3 prompt."}),
                "model": (_model_dropdown(), {
                    "default": OR_DEFAULT_MODEL,
                    "tooltip": "OpenRouter model slug. List wird live von der "
                               "OpenRouter-API geladen (text-only Modelle). Refresh "
                               "ComfyUI, um neu hinzugekommene Modelle zu sehen."}),
                "skill": (_list_skills(), {
                    "default": "none",
                    "tooltip": "Prompting skill that steers the LLM. 'none' = freeform. "
                               "Bundled: Flux3Director, Flux3Director4Discord. "
                               "Drop extra .md files into the skills/ folder and "
                               "refresh ComfyUI to add your own."}),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Optional reference image(s) for the LLM to see — "
                               "the model must be vision-capable (e.g. GPT-4o, "
                               "Claude Sonnet, Gemini). The image is sent as base64 "
                               "alongside your idea; describe it in the idea text."}),
                "model_custom": ("STRING", {
                    "default": "",
                    "tooltip": "Optional: beliebiger OpenRouter-Model-Slug, der nicht "
                               "im Dropdown steht. Hat Vorrang vor dem model-Dropdown."}),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Leer = aus .env / Umgebungsvariable "
                               "OPENROUTER_API_KEY."}),
                "extra_instructions": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional: extra directives appended to the skill "
                               "(z.B. 'make it 10s, 2 segments, dialogue in German')."}),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LLM sampling temperature. Lower = deterministic, "
                               "higher = creative."}),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 256, "max": 32000,
                    "tooltip": "Max output tokens."}),
                "timeout_seconds": ("INT", {
                    "default": 120, "min": 10, "max": 600,
                    "tooltip": "Wie lange die Node auf die LLM-Antwort wartet."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    async def generate(self, idea, model, skill,
                       model_custom="", api_key="", extra_instructions="",
                       temperature=0.7, max_tokens=4096, timeout_seconds=120,
                       images=None, **kwargs):
        if not idea.strip():
            raise ValueError("Flux3Prompt: idea darf nicht leer sein.")

        key = get_openrouter_api_key(api_key)

        custom = (model_custom or "").strip()
        if custom == "(provider default)":
            custom = ""
        chosen = custom if custom else model

        image_blobs: list[str] | None = None
        if images is not None:
            if not is_vision_model(chosen):
                log.warning("Flux3Prompt: model '%s' unterstützt vermutlich keine "
                            "Bilder. Nutze ein Vision-Modell für Reference-Images.", chosen)
            image_blobs = await asyncio.to_thread(
                lambda imgs: [image_to_base64(im) for im in imgs], images)
            log.info("Flux3Prompt: %d reference image(s) attached.", len(image_blobs))

        skill_text = _load_skill(skill) if skill and skill != "none" else ""
        parts = [_PROMPTER_SYSTEM_FRAME]
        if skill_text:
            parts.append("--- PROMPTING SKILL ---\n" + skill_text)
        else:
            parts.append("(No skill selected — produce a clean, vivid FLUX 3 prompt.)")
        if extra_instructions.strip():
            parts.append("--- ADDITIONAL INSTRUCTIONS ---\n" + extra_instructions.strip())
        if image_blobs:
            parts.append(
                "--- REFERENCE IMAGES ---\n"
                "One or more reference images are attached to the user message. "
                "Describe how they should inform the FLUX 3 prompt — e.g. which "
                "image defines a character's appearance, which defines a style, "
                "which is a keyframe. Reference them in the Cast / Setting / Look "
                "sections using the image's visual content.")
        system = "\n\n".join(parts)

        user = f"Video idea:\n{idea.strip()}"

        log.info("Flux3Prompt: calling OpenRouter (model=%s, skill=%s, %.0f tokens max, images=%s)",
                 chosen, skill, max_tokens, len(image_blobs) if image_blobs else 0)

        text = await asyncio.to_thread(
            call_openrouter, key, chosen, system, user,
            temperature, max_tokens, timeout_seconds, image_blobs,
        )

        if not text:
            raise RuntimeError("Flux3Prompt: LLM returned an empty response.")
        return (text,)


NODE_CLASS_MAPPINGS = {
    "Flux3VideoDRE": Flux3Video,
    "Flux3VideoUpscaleDRE": Flux3VideoUpscale,
    "Flux3PrompterDRE": Flux3Prompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux3VideoDRE": "🎬 Flux 3 Video (API) *DRE",
    "Flux3VideoUpscaleDRE": "🎬 Flux 3 Video Upscale (API) *DRE",
    "Flux3PrompterDRE": "🎬 Flux 3 Openrouter Prompt *DRE",
}
