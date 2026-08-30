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
            f"Flux3: keyframe_times must be a list of seconds (e.g. '0, 4.5, 10'), "
            f"got: {text!r}"
        ) from exc


async def _build_keyframes(images: torch.Tensor | None, end_image: torch.Tensor | None,
                           keyframe_times: str, duration_value: str) -> Any:
    """Build the `keyframes` payload for i2v per the documented schema.

    - 1 image, no times: opening frame           -> str
    - 2 images (or images + end_image): start+end -> [str, str]
    - 3+ images, no times: evenly spread          -> [str, ...]  (duration must be set)
    - images + keyframe_times: storyboard         -> [[seconds, str], ...]
    """
    if images is None:
        raise ValueError("Flux3: i2v needs an images input.")

    # images + end_image is the convenient way to pin start and end.
    if end_image is not None:
        imgs = torch.cat([images[:1], end_image[:1]])
    else:
        imgs = images

    n = len(imgs)
    if n < 1:
        raise ValueError("Flux3: i2v needs at least one image on the images input.")
    if n > MAX_KEYFRAMES:
        raise ValueError(
            f"Flux3: i2v accepts at most {MAX_KEYFRAMES} keyframes, but the batch has {n}."
        )

    times_text = keyframe_times.strip()
    if times_text:
        times = _parse_keyframe_times(times_text)
        if len(times) != n:
            raise ValueError(
                f"Flux3: keyframe_times needs as many values as there are images - "
                f"{n} image(s) but {len(times)} time(s)."
            )
        if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
            raise ValueError(
                f"Flux3: keyframe_times must ascend, got: {times}"
            )
        if any(t < 0 for t in times):
            raise ValueError(f"Flux3: keyframe_times must not be negative: {times}")
        encoded = await asyncio.to_thread(batch_to_base64, imgs)
        return [[t, b64] for t, b64 in zip(times, encoded)]

    if n >= 3 and duration_value == "auto":
        # Spec: "3 or more need a set duration."
        raise ValueError(
            "Flux3: 3+ keyframes without keyframe_times need a fixed duration (5-20). Set duration "
            "to a value, or supply keyframe_times."
        )

    encoded = await asyncio.to_thread(batch_to_base64, imgs)
    if n == 1:
        return encoded[0]
    return encoded  # list[str]: 2 -> start+end, 3+ -> evenly spread


class Flux3Video:
    """FLUX 3 Video - t2v / i2v / v2v / draft_enhance, with synchronised audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (VIDEO_MODES, {
                    "default": "t2v",
                    "tooltip": "t2v: text to video. i2v: image(s) to video (keyframes). v2v: "
                               "continue a clip (start_video). draft_enhance: render an earlier "
                               "draft at full quality."}),
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Required for t2v/i2v/v2v. Not allowed with draft_enhance, where "
                               "the draft_cache pins everything."}),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "auto",
                    "tooltip": "'auto' lets the API choose based on the prompt and the "
                               "references."}),
                "duration": (DURATIONS, {
                    "default": "auto",
                    "tooltip": "Whole seconds 5-20, or 'auto'. 3+ keyframes without keyframe_times "
                               "need a fixed length."}),
                "resolution": (RESOLUTIONS, {
                    "default": "hd",
                    "tooltip": "hd = default. fhd = higher resolution, via the video "
                               "upscaler."}),
                "generate_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On by default. Off gives a silent clip."}),
                "safety_tolerance": ("INT", {
                    "default": SAFETY_TOLERANCE_DEFAULT,
                    "min": SAFETY_TOLERANCE_MIN, "max": SAFETY_TOLERANCE_MAX,
                    "tooltip": "0 (strictest) to 4. Default 2. With conditioning media the maximum "
                               "is 2."}),
                "draft": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Fast hd preview. The result carries a draft_cache that you can "
                               "render at full quality with mode=draft_enhance."}),
                "version": ("STRING", {
                    "default": "latest",
                    "tooltip": "Only 'latest' is available at the moment."}),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "i2v only. 1 image is the start frame, 2 images are start and end, "
                               "3-10 images are spaced evenly and need duration. Or combine with "
                               "keyframe_times for a storyboard."}),
                "end_image": ("IMAGE", {
                    "tooltip": "i2v only: when set, images is the start and end_image the end, "
                               "giving exactly 2 keyframes."}),
                "keyframe_times": ("STRING", {
                    "default": "",
                    "tooltip": "i2v only: seconds per image, comma separated (e.g. '0, 4.5, 10'). "
                               "The count must match images and the values must ascend. Each image "
                               "becomes a frame at that second."}),
                "video": ("VIDEO", {
                    "tooltip": "v2v only: the clip to continue (mp4)."}),
                "draft_cache": ("STRING", {
                    "default": "",
                    "tooltip": "draft_enhance only: the base64 bundle or URL from the draft output "
                               "of an earlier run. Required in this mode."}),
                "timeout_minutes": ("INT", {
                    "default": DEFAULT_TIMEOUT_MINUTES, "min": 1, "max": 240,
                    "tooltip": "How long the node waits for the result. With a busy BFL queue a "
                               "job easily takes 20+ minutes. When the time runs out only the node "
                               "stops; the job keeps running server-side and still costs credits."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Leave empty to read it from "
                                                                 ".env"}),
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
                f"Flux3: mode '{mode}' is not supported. Allowed: {', '.join(VIDEO_MODES)}.")

        # --- draft_enhance: only mode, draft_cache, safety_tolerance are accepted.
        if mode == "draft_enhance":
            cache = draft_cache.strip()
            if not cache:
                raise ValueError(
                    "Flux3: draft_enhance needs draft_cache, a base64 bundle or URL from an "
                    "earlier draft run."
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
                log.warning("Flux3: draft_enhance only accepts draft_cache + safety_tolerance. "
                            "These inputs are ignored: %s", ", ".join(ignored))
        else:
            if not prompt.strip():
                raise ValueError("Flux3: prompt must not be empty.")

            payload: dict = {"mode": mode, "prompt": prompt.strip()}

            if aspect_ratio not in ASPECT_RATIOS:
                raise ValueError(
                    f"Flux3: aspect_ratio '{aspect_ratio}' is not supported. "
                    f"Allowed: {', '.join(ASPECT_RATIOS)}."
                )
            payload["aspect_ratio"] = aspect_ratio

            # duration: "auto" oder int 5..20
            if duration not in DURATIONS:
                raise ValueError(
                    f"Flux3: duration '{duration}' is not supported. "
                    f"Allowed: auto, or whole seconds 5-20."
                )
            payload["duration"] = duration if duration == "auto" else int(duration)

            if resolution not in RESOLUTIONS:
                raise ValueError(
                    f"Flux3: resolution '{resolution}' is not supported. "
                    f"Allowed: {', '.join(RESOLUTIONS)}."
                )
            payload["resolution"] = resolution

            payload["generate_audio"] = bool(generate_audio)

            if not (SAFETY_TOLERANCE_MIN <= safety_tolerance <= SAFETY_TOLERANCE_MAX):
                raise ValueError(
                    f"Flux3: safety_tolerance must be "
                    f"{SAFETY_TOLERANCE_MIN}..{SAFETY_TOLERANCE_MAX}, got: {safety_tolerance}."
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
                    raise ValueError("Flux3: v2v needs a video input (start_video).")
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
                    "tooltip": "The clip to upscale (mp4). Max "
                               f"{UPSCALE_VIDEO_MAX_SECONDS} s and "
                               f"{UPSCALE_VIDEO_MAX_MB} MB. Or use "
                               "input_video_url instead."}),
                "target_resolution": (UPSCALE_TARGETS, {
                    "default": "4K",
                    "tooltip": "Target resolution on the short side. The node measures the source "
                               "and derives upscale_factor from it (factor = target_short / min(w, "
                               "h), clamped to 1.5-3.0), keeping the source aspect ratio. 1080p is "
                               "about 1920x1080, 2K about 2560x1440, 4K about 3840x2160, each at "
                               "16:9."}),
                "creativity": (UPSCALE_CREATIVITY_MODES, {
                    "default": "creative",
                    "tooltip": "precise (0) preserves identity and stays sharp, for faces, "
                               "products and real people. creative (1, default) invents detail, "
                               "good for generated footage, landscapes and textures, but faces and "
                               "products may drift."}),
            },
            "optional": {
                "input_video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Alternative to the video input: an HTTP(S) URL to the source. It "
                               "takes precedence over the video input and avoids base64 encoding a "
                               "large clip. The node only streams the moov atom of the URL to "
                               "measure the source dimensions, so there is no full download."}),
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional description of the clip. It steers the "
                               "enhanced detail, above all in creative mode. "
                               "Leave empty for a neutral upscale."}),
                "safety_tolerance": ("INT", {
                    "default": SAFETY_TOLERANCE_DEFAULT,
                    "min": SAFETY_TOLERANCE_MIN, "max": SAFETY_TOLERANCE_MAX,
                    "tooltip": "0 (strictest) to 4, default 2. Moderates the prompt and the "
                               "delivered frames."}),
                "timeout_minutes": ("INT", {
                    "default": DEFAULT_TIMEOUT_MINUTES, "min": 1, "max": 240,
                    "tooltip": "How long the node waits for the result. An upscale can take about "
                               "as long as generating a video."}),
                "api_key": ("STRING", {"default": "", "tooltip": "Leave empty to read it from "
                                                                 ".env"}),
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
        raise ValueError(f"Flux3: no video stream in {url}")

    async def generate(self, video, target_resolution, creativity,
                       input_video_url="", prompt="",
                       safety_tolerance=SAFETY_TOLERANCE_DEFAULT,
                       timeout_minutes=DEFAULT_TIMEOUT_MINUTES, api_key=""):
        if target_resolution not in UPSCALE_TARGET_SHORT_SIDE:
            raise ValueError(
                f"Flux3: target_resolution '{target_resolution}' is not supported. "
                f"Allowed: {', '.join(UPSCALE_TARGETS)}."
            )
        target_short = UPSCALE_TARGET_SHORT_SIDE[target_resolution]

        # --- Source dimensions: URL takes priority; otherwise the VIDEO input.
        url = (input_video_url or "").strip()
        if url:
            if not url.startswith("http"):
                raise ValueError(
                    "Flux3: input_video_url must be an HTTP(S) URL."
                )
            src_w, src_h = await asyncio.to_thread(
                self._probe_url_dimensions, url)
            input_video = url
        else:
            if video is None:
                raise ValueError(
                    "Flux3: upscale needs either the video input or input_video_url."
                )
            src_w, src_h = video.get_dimensions()
            input_video = await asyncio.to_thread(video_to_base64, video)

        # --- Compute the upscale_factor from the source short side.
        source_short = min(src_w, src_h)
        factor = target_short / source_short
        if factor < UPSCALE_FACTOR_MIN:
            raise ValueError(
                f"Flux3: the source is {src_w}x{src_h} (short side {source_short}px) "
                f"and therefore already at or above {target_resolution} (short side "
                f"{target_short}px). Upscaling would give a factor of "
                f"{factor:.2f} < min {UPSCALE_FACTOR_MIN}, so there is nothing to upscale. "
                f"Pick a higher target_resolution (e.g. 4K) or a "
                f"smaller source."
            )
        if factor > UPSCALE_FACTOR_MAX:
            log.warning(
                "Flux3: %s is not reachable from a %dx%d source (short side %d) - the factor would "
                "be %.2f, max %.1f. Upscaling with %.1fx instead, effective short side ~%dpx.",
                target_resolution, src_w, src_h, source_short,
                factor, UPSCALE_FACTOR_MAX, UPSCALE_FACTOR_MAX,
                int(source_short * UPSCALE_FACTOR_MAX),
            )
            factor = UPSCALE_FACTOR_MAX

        if creativity not in UPSCALE_CREATIVITY_MODES:
            raise ValueError(
                f"Flux3: creativity '{creativity}' is not supported. "
                f"Allowed: {', '.join(UPSCALE_CREATIVITY_MODES)}."
            )
        creativity_val = 1 if creativity == "creative" else 0

        if not (SAFETY_TOLERANCE_MIN <= safety_tolerance <= SAFETY_TOLERANCE_MAX):
            raise ValueError(
                f"Flux3: safety_tolerance must be {SAFETY_TOLERANCE_MIN}..{SAFETY_TOLERANCE_MAX}, "
                f"got: {safety_tolerance}."
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
        raise FileNotFoundError(f"Flux3Prompt: skill '{name}' not found in {_SKILLS_DIR}.")
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
                    "tooltip": "OpenRouter model slug. The list is loaded live from the OpenRouter "
                               "API (text-only models). Refresh ComfyUI to pick up newly added "
                               "models."}),
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
                    "tooltip": "Optional: any OpenRouter model slug that is not in the dropdown. "
                               "Takes precedence over the model dropdown."}),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Leave empty to read it from .env or the "
                               "OPENROUTER_API_KEY environment variable."}),
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
                    "tooltip": "How long the node waits for the LLM response."}),
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
            raise ValueError("Flux3Prompt: idea must not be empty.")

        key = get_openrouter_api_key(api_key)

        custom = (model_custom or "").strip()
        if custom == "(provider default)":
            custom = ""
        chosen = custom if custom else model

        image_blobs: list[str] | None = None
        if images is not None:
            if not is_vision_model(chosen):
                log.warning("Flux3Prompt: model '%s' probably does not support images. Use a vision "
                            "model for reference images.", chosen)
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

        log.info("Flux3Prompt: calling OpenRouter (model=%s, skill=%s, %.0f tokens max, "
                 "images=%s)",
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
