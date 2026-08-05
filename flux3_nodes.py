"""ComfyUI nodes for the BFL Flux 3 API.

Flux3Video — documented endpoint POST /v1/flux-3-video, four modes
(t2v / i2v / v2v / draft_enhance).
Spec: https://docs.bfl.ai/api-reference/utility/generate-a-video-with-flux-3

Flux3Image — PRE-RELEASE: the image API is not documented yet
(https://docs.bfl.ai/flux_3/flux3_image says "Endpoint slugs and pricing are TBA
until launch"). This node talks to the probed flux-3-large/distilled/small
endpoints so it's ready when BFL flips the switch. Lives on the dev branch only.

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
    IMAGE_ASPECT_RATIOS,
    IMAGE_MODELS,
    IMAGE_SIZES,
    MAX_KEYFRAMES,
    MAX_SEED,
    RESOLUTIONS,
    SAFETY_TOLERANCE_DEFAULT,
    SAFETY_TOLERANCE_MAX,
    SAFETY_TOLERANCE_MIN,
    VIDEO_MODES,
    Flux3Client,
    batch_to_base64,
    bytes_to_image_tensor,
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
               timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
               image_model: str = "") -> tuple[dict, dict, bytes]:
    """Submit, poll and download without blocking the executor.

    Every step is either awaited or pushed onto a thread, so ComfyUI can run other
    Flux nodes in the same graph concurrently instead of one after another.

    image_model: when set, submit to that pre-release image endpoint instead of the
    documented video endpoint.
    """
    task = await asyncio.to_thread(client.submit, payload, image_model)
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


class Flux3Image:
    """Text-to-image (t2i) and multi-reference edit (i2i).

    PRE-RELEASE: the image API isn't documented yet (docs say slugs/pricing are
    TBA until launch). This node uses the probed flux-3-large/distilled/small
    endpoints. Lives on the dev branch only; not shipped on main.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (IMAGE_MODELS, {
                    "default": "flux-3-large",
                    "tooltip": "large = volle Qualität · distilled = schnell · "
                               "small = klein/günstig (gleiches Schema). Pre-release; "
                               "Slug kann sich zum Launch noch ändern."}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional; Feld wird nur gesendet, wenn nicht leer."}),
                # Capped at 2^32-1: the documented video endpoint rejects anything
                # above that, and ComfyUI's "randomize" would otherwise produce a
                # failing seed nearly every run. Safe for the image endpoints too.
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED,
                                 "tooltip": "0 = zufällig (Feld wird nicht gesendet)."}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 50,
                                  "tooltip": "1-50. 0 = API-Default verwenden"}),
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1,
                                       "tooltip": "0 = API-Default verwenden"}),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
                "aspect_ratio": (IMAGE_ASPECT_RATIOS, {"default": "16:9"}),
                "image_size": (IMAGE_SIZES, {
                    "default": "1024sq",
                    "tooltip": "Größe als Quadrat-Äquivalent; aspect_ratio bestimmt die Form."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "Angeschlossen = i2i (Edit/Remix). "
                                                "Bis zu 4 Bilder aus einem Batch."}),
                "alpha": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 100.0, "step": 0.01,
                    "tooltip": "Undokumentierter Parameter der API (float). "
                               "-1 = nicht senden / API-Default."}),
                "conditioning_noise": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Nur i2i (images angeschlossen): Rauschen auf dem "
                               "Eingangsbild (>= 0). -1 = nicht senden / API-Default."}),
                "timeout_minutes": ("INT", {
                    "default": DEFAULT_TIMEOUT_MINUTES, "min": 1, "max": 240,
                    "tooltip": "Wie lange die Node auf das Ergebnis wartet."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Leer = aus .env"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    async def generate(self, model, prompt, negative_prompt, seed, steps, guidance,
                       prompt_upsampling, aspect_ratio, image_size, images=None,
                       alpha=-1.0, conditioning_noise=-1.0,
                       timeout_minutes=DEFAULT_TIMEOUT_MINUTES, api_key=""):
        if not prompt.strip():
            raise ValueError("Flux3: prompt darf nicht leer sein.")

        mode = "i2i" if images is not None else "t2i"
        payload: dict = {"mode": mode, "prompt": prompt.strip()}
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()
        if seed:
            if seed > MAX_SEED:
                raise ValueError(
                    f"Flux3: seed darf höchstens {MAX_SEED} sein, bekommen: {seed}."
                )
            payload["seed"] = int(seed)
        if steps:
            payload["steps"] = int(steps)
        if guidance:
            payload["guidance"] = float(guidance)
        if prompt_upsampling:
            payload["prompt_upsampling"] = True
        payload["aspect_ratio"] = aspect_ratio
        payload["image_size"] = image_size
        if alpha >= 0:
            payload["alpha"] = float(alpha)
        # conditioning_noise exists on i2i (probed 2026-07-26), not on t2i.
        if conditioning_noise >= 0 and mode == "i2i":
            payload["conditioning_noise"] = float(conditioning_noise)

        if images is not None:
            encoded = await asyncio.to_thread(batch_to_base64, images)
            payload["input_image"] = encoded if len(encoded) > 1 else encoded[0]

        client = Flux3Client(api_key)
        task, result, data = await _run(client, payload, timeout_minutes,
                                        image_model=model)
        return (bytes_to_image_tensor(data),
                format_metadata(payload, result, task, image_model=model))


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
    "Flux3PrompterDRE": Flux3Prompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux3VideoDRE": "🎬 Flux 3 Video (API)",
    "Flux3PrompterDRE": "🎬 Flux 3 Openrouter Prompt",
}
