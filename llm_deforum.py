"""LLM-powered Deforum schedule generator node.

Uses the existing llm_client.py (OpenRouter) to generate all 7 Deforum
schedules in one API call:
  - Prompt Schedule (deforum-comfy-nodes format: "frame": "prompt",)
  - Denoise Schedule  (format: 0:(0.4),50:(0.5),)
  - Seed Schedule     (format: 0:(42),50:(137),)
  - Zoom Schedule     (format: 0:(1.0),40:(1.02),)
  - X Translation     (format: 0:(0),40:(2.0),)
  - Y Translation     (format: 0:(0),40:(1.5),)
  - Rotation          (format: 0:(0),40:(1.0),)

The LLM is instructed to return a JSON object with these 7 keys; the output
is validated and fixed up (trailing commas, missing 0: keyframe, etc.) so it
can be plugged directly into deforum-comfy-nodes ValueSchedule / PromptSchedule
nodes.
"""

import json
import re
import time

from .llm_client import fetch_openrouter_models, get_api_key, call_openrouter


SYSTEM_PROMPT_TEMPLATE = """You are a Deforum animation schedule generator. Given a user's concept and a total number of frames, you generate 7 schedules for a Deforum animation pipeline.

Output ONLY a valid JSON object with exactly these 7 string keys. No markdown, no code fences, no explanation — just the JSON object.

KEYS AND THEIR FORMAT RULES:

1. "prompt_schedule": Frame-keyed prompts. Format: "frame": "prompt text", (with trailing comma). Frames should be evenly distributed across the total frame count. Each entry on its own line. Example for 200 frames:
"0": "A teacup filled with swirling galaxies, porcelain cracking",
"40": "The teacup grows legs of twisted coral",
"80": "The teacup reaches a mountain of sleeping faces",
"120": "The teacup explodes into a flock of porcelain birds",
"160": "The birds dissolve into a single enormous tongue",
"200": "Everything collapses into a drop of liquid universe"

2. "denoise_schedule": Value schedule. Format: frame:(value), (with trailing comma). Values 0.0-1.0. Lower = more coherent, higher = more change. For stable animation keep 0.3-0.6. Example:
0:(0.4),50:(0.5),100:(0.45),150:(0.5),200:(0.4),

3. "seed_schedule": Value schedule. Integer seeds that change at keyframe boundaries. Format: frame:(seed), (with trailing comma). Example:
0:(42),50:(137),100:(89),150:(256),200:(42),

4. "zoom_schedule": Value schedule. 1.0 = no zoom, >1.0 = zoom in, <1.0 = zoom out. Space keyframes AT LEAST 20 frames apart. Format: frame:(value), (with trailing comma). Example for 200 frames:
0:(1.0),40:(1.02),80:(1.0),120:(0.98),160:(1.02),200:(1.0),

5. "x_translation_schedule": Value schedule. Positive = right, negative = left. Range -10 to 10. Space keyframes AT LEAST 20 frames apart. Format: frame:(value), (with trailing comma). Example:
0:(0),40:(2.0),80:(0),120:(-2.0),160:(0),200:(0),

6. "y_translation_schedule": Value schedule. Positive = down, negative = up. Range -10 to 10. Space keyframes AT LEAST 20 frames apart. Format: frame:(value), (with trailing comma). Example:
0:(0),40:(1.5),80:(0),120:(-1.5),160:(0),200:(0),

7. "rotation_schedule": Value schedule. Rotation in degrees. Range -5 to 5. Space keyframes AT LEAST 20 frames apart. Format: frame:(value), (with trailing comma). Example:
0:(0),40:(1.0),80:(0),120:(-1.0),160:(0),200:(0),

PROMPT SPACING RULE (CRITICAL):
- Space prompts AT LEAST 20 frames apart. Never place prompts closer than 20 frames.
- Maximum number of prompt keyframes = total_frames / 20, rounded down, minimum 2 (frame 0 and last frame).
- Examples:
  - 10 frames → 2 prompts: "0", "10"
  - 20 frames → 2 prompts: "0", "20"
  - 50 frames → 3 prompts: "0", "25", "50"
  - 100 frames → 5 prompts: "0", "25", "50", "75", "100"
  - 200 frames → 10 prompts: "0", "20", "40", "60", "80", "100", "120", "140", "160", "200"

RULES:
- Distribute keyframes evenly across the given total frames.
- Keep motion schedules subtle for coherent animation (small values).
- The prompt schedule should tell a continuous evolving story.
- Make the animation visually interesting but not chaotic.
- All value schedules MUST end with a trailing comma.
- The prompt schedule entries MUST be separated by newlines.
- Output ONLY the JSON object, nothing else.

STYLE HINT: {style_hint}

USER CONCEPT: {user_prompt}
TOTAL FRAMES: {num_frames}

Generate the 7 schedules now. Output only the JSON object."""


def _build_system_prompt(user_prompt, num_frames, style_hint):
    return SYSTEM_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt,
        num_frames=num_frames,
        style_hint=style_hint,
    )


def _extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*}", "}", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _validate_prompt_schedule(text, num_frames):
    # If LLM returned a dict (nested JSON object), convert to string format
    if isinstance(text, dict):
        def _key_num(k):
            try:
                return int(k)
            except (ValueError, TypeError):
                return 0
        lines = []
        for key in sorted(text.keys(), key=_key_num):
            lines.append(f'"{key}": "{text[key]}"')
        text = "\n".join(lines)

    if not text or not isinstance(text, str):
        return _default_prompt_schedule(num_frames)

    text = text.strip()

    # If the whole thing is a JSON string like '{"0": "cat", "30": "dog"}',
    # try to parse it as JSON first, then convert the dict back to lines
    if text.startswith("{") and text.endswith("}"):
        try:
            inner = json.loads(text)
            if isinstance(inner, dict):
                def _key_num2(k):
                    try:
                        return int(k)
                    except (ValueError, TypeError):
                        return 0
                lines = []
                for key in sorted(inner.keys(), key=_key_num2):
                    lines.append(f'"{key}": "{inner[key]}"')
                text = "\n".join(lines)
        except json.JSONDecodeError:
            pass

    # Strip any remaining outer braces
    while text.startswith("{"):
        text = text[1:].strip()
    while text.endswith("}"):
        text = text[:-1].strip()

    # Remove standalone brace lines and strip stray braces from each line
    lines = [l.strip() for l in text.split("\n") if l.strip() and l.strip() not in ("{", "}")]

    fixed = []
    for line in lines:
        line = line.strip()
        # Remove ALL stray braces from each line (handles {"key": "val"}, etc.)
        line = line.replace("{", "").replace("}", "").strip()
        if not line:
            continue
        if not line.endswith(","):
            line = line + ","
        fixed.append(line)

    if not fixed:
        return _default_prompt_schedule(num_frames)

    result = "\n".join(fixed)

    # Final safety: if the result still starts with { (shouldn't happen), bail
    if result.strip().startswith("{"):
        return _default_prompt_schedule(num_frames)

    return result


def _validate_value_schedule(text, num_frames, default_value=0.0):
    if not text or not isinstance(text, str):
        return _default_value_schedule(num_frames, default_value)
    text = text.strip()
    if not text.endswith(","):
        text = text + ","
    if "0:" not in text:
        text = f"0:({default_value})," + text
    return text


def _default_prompt_schedule(num_frames):
    return (
        f'"0": "A psychedelic dreamscape begins to form from static noise",\n'
        f'"{num_frames // 2}": "The dreamscape reaches peak intensity and complexity",\n'
        f'"{num_frames}": "The dreamscape dissolves back into calm stillness",'
    )


def _default_value_schedule(num_frames, default_value=0.0):
    quarter = max(1, num_frames // 4)
    half = max(1, num_frames // 2)
    three_q = max(1, num_frames * 3 // 4)
    return (
        f"0:({default_value}),"
        f"{quarter}:({default_value}),"
        f"{half}:({default_value}),"
        f"{three_q}:({default_value}),"
        f"{num_frames}:({default_value}),"
    )


class LLMDeforumGenerator:
    """LLM-powered Deforum schedule generator (OpenRouter).

    Generates all 7 Deforum schedules in one LLM call. Outputs are strings
    in the exact format expected by deforum-comfy-nodes ValueSchedule and
    PromptSchedule nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (fetch_openrouter_models(),),
                "num_frames": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 9999,
                    "step": 1,
                }),
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A psychedelic journey through a cat's dream, morphing through cosmic landscapes",
                }),
                "style": ([
                    "psychedelic",
                    "horror",
                    "dreamy",
                    "abstract",
                    "cinematic",
                    "surreal",
                    "organic",
                    "geometric",
                    "noir",
                    "vaporwave",
                ], {"default": "psychedelic"}),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "round": 1,
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                }),
            },
            "optional": {
                "custom_system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "Prompt Schedule | Deforum",
        "Denoise Schedule",
        "Seed Schedule",
        "Zoom Schedule",
        "X Translation Schedule",
        "Y Translation Schedule",
        "Rotation Schedule",
        "Stats",
    )

    FUNCTION = "generate"
    CATEGORY = "denrakeiw/LLM"

    def generate(self, model, num_frames, user_prompt, style, temperature, api_key,
                 custom_system_prompt=""):
        stats_parts = []

        if not user_prompt.strip():
            stats_parts.append("Error: No user prompt provided.")
            return self._fallback_output(num_frames, "\n".join(stats_parts))

        try:
            resolved_key = get_api_key(api_key)
        except RuntimeError as e:
            stats_parts.append(str(e))
            return self._fallback_output(num_frames, "\n".join(stats_parts))

        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        else:
            system_prompt = _build_system_prompt(user_prompt, num_frames, style)

        user_message = (
            f"Generate all 7 Deforum schedules for a {num_frames}-frame animation. "
            f"Style: {style}. Concept: {user_prompt}"
        )

        try:
            start_time = time.time()
            raw_response = call_openrouter(
                api_key=resolved_key,
                model=model,
                system=system_prompt,
                user=user_message,
                temperature=temperature,
                max_tokens=4096,
            )
            elapsed = time.time() - start_time
        except Exception as e:
            stats_parts.append(f"API error: {e}")
            return self._fallback_output(num_frames, "\n".join(stats_parts))

        if not raw_response:
            stats_parts.append("API returned empty response.")
            return self._fallback_output(num_frames, "\n".join(stats_parts))

        parsed = _extract_json(raw_response)
        if parsed is None:
            stats_parts.append(f"Failed to parse JSON. Raw (first 300 chars): {raw_response[:300]}")
            return self._fallback_output(num_frames, "\n".join(stats_parts))

        prompt_sched = _validate_prompt_schedule(
            parsed.get("prompt_schedule", ""), num_frames
        )
        denoise_sched = _validate_value_schedule(
            parsed.get("denoise_schedule", ""), num_frames, 0.4
        )
        seed_sched = _validate_value_schedule(
            parsed.get("seed_schedule", ""), num_frames, 42
        )
        zoom_sched = _validate_value_schedule(
            parsed.get("zoom_schedule", ""), num_frames, 1.0
        )
        x_sched = _validate_value_schedule(
            parsed.get("x_translation_schedule", ""), num_frames, 0.0
        )
        y_sched = _validate_value_schedule(
            parsed.get("y_translation_schedule", ""), num_frames, 0.0
        )
        rot_sched = _validate_value_schedule(
            parsed.get("rotation_schedule", ""), num_frames, 0.0
        )

        stats_parts.append(f"Model: {model}")
        stats_parts.append(f"Elapsed: {elapsed:.2f}s")
        stats_parts.append(f"Frames: {num_frames}")
        stats_parts.append("OK")

        return (
            prompt_sched,
            denoise_sched,
            seed_sched,
            zoom_sched,
            x_sched,
            y_sched,
            rot_sched,
            "\n".join(stats_parts),
        )

    def _fallback_output(self, num_frames, stats):
        return (
            _default_prompt_schedule(num_frames),
            _default_value_schedule(num_frames, 0.4),
            _default_value_schedule(num_frames, 42),
            _default_value_schedule(num_frames, 1.0),
            _default_value_schedule(num_frames, 0.0),
            _default_value_schedule(num_frames, 0.0),
            _default_value_schedule(num_frames, 0.0),
            stats,
        )

    @classmethod
    def IS_CHANGED(cls, model, num_frames, user_prompt, style, temperature, api_key,
                   custom_system_prompt=""):
        return (model, num_frames, user_prompt, style, float(temperature),
                api_key, custom_system_prompt)


NODE_CLASS_MAPPINGS = {
    "LLMDeforumGenerator": LLMDeforumGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMDeforumGenerator": "🤖 LLM Deforum (WIP)",
}
