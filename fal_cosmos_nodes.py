"""ComfyUI node for the NVIDIA Cosmos 3 Super text-to-image endpoint on fal.ai.

Cosmos3SuperTextToImage — POST queue.fal.run/nvidia/cosmos-3-super/text-to-image,
queue/polling via raw REST (see fal_client.py).
Spec: https://fal.ai/models/nvidia/cosmos-3-super/text-to-image/api
"""

import asyncio
import logging

import torch

from .fal_client import (
    DEFAULT_TIMEOUT_MINUTES,
    ENDPOINT_T2I,
    FalClient,
    format_metadata,
)
from .flux3_client import bytes_to_image_tensor

log = logging.getLogger("FalAPI")

CATEGORY = "fal.ai API"

IMAGE_SIZES = [
    "square_hd", "square",
    "portrait_4_3", "portrait_16_9",
    "landscape_4_3", "landscape_16_9",
    "custom",
]
OUTPUT_FORMATS = ["jpeg", "png"]
MAX_SEED = 4294967295

# fal clamps each edge to 512-1280px (multiples of 16).
SIZE_MIN = 512
SIZE_MAX = 1280
SIZE_STEP = 16


class Cosmos3SuperTextToImage:
    """NVIDIA Cosmos 3 Super Text-to-Image via fal.ai."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Beschreibung des Bildes. Cosmos3 wurde auf dichte, "
                               "strukturierte Captions trainiert — ausführliche Prompts "
                               "funktionieren besser."}),
                "image_size": (IMAGE_SIZES, {
                    "default": "square_hd",
                    "tooltip": "fal-Preset oder 'custom' für eigene width/height "
                               "(512–1280, Schritte von 16)."}),
                "width": ("INT", {
                    "default": 1024, "min": SIZE_MIN, "max": SIZE_MAX, "step": SIZE_STEP,
                    "tooltip": "Nur bei image_size='custom'."}),
                "height": ("INT", {
                    "default": 1024, "min": SIZE_MIN, "max": SIZE_MAX, "step": SIZE_STEP,
                    "tooltip": "Nur bei image_size='custom'."}),
                "num_inference_steps": ("INT", {
                    "default": 28, "min": 1, "max": 50,
                    "tooltip": "Denoising-Steps. Mehr = höhere Qualität, langsamer."}),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance. Höher = strengere Prompt-Treue."}),
                "num_images": ("INT", {
                    "default": 1, "min": 1, "max": 4,
                    "tooltip": "Anzahl Bilder pro Run (werden als Batch ausgegeben)."}),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": MAX_SEED,
                    "control_after_generate": True,
                    "tooltip": "Gleiche Seed + Prompt + Modellversion = gleiches Bild."}),
                "output_format": (OUTPUT_FORMATS, {
                    "default": "jpeg",
                    "tooltip": "jpeg (kleiner) oder png (verlustfrei)."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Wovon die Generation weggesteuert wird "
                               "(Farben, Objekte, Artefakte). Leer = aus."}),
                "enable_prompt_expansion": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "LLM schreibt den Prompt vorab in das dichte "
                               "Cosmos3-Trainingsformat um. Bei Fehler fällt die API "
                               "auf den Rohtext zurück."}),
                "enable_agentic_generation": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Iterativer Agentic-Loop: mehrere Kandidaten generieren, "
                               "bewerten, Prompt nachschärfen. Deutlich langsamer und "
                               "teurer (jeder Kandidat ist eine volle Generation)."}),
                "agentic_max_iterations": ("INT", {
                    "default": 2, "min": 1, "max": 5,
                    "tooltip": "Nur bei Agentic: maximale Verfeinerungsrunden."}),
                "agentic_samples_per_iteration": ("INT", {
                    "default": 2, "min": 1, "max": 4,
                    "tooltip": "Nur bei Agentic: Kandidaten pro Runde."}),
                "agentic_early_stop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Nur bei Agentic: Loop früh stoppen, wenn ein Kandidat "
                               "die Qualitätsschwelle reißt."}),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Content-Moderation für Prompt und Ergebnis. "
                               "Deaktivieren braucht Account-Freigabe bei fal."}),
                "sync_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True = Bild kommt als data-URI zurück und taucht nicht "
                               "in der fal-History auf. Fast nie nötig."}),
                "timeout_minutes": ("INT", {
                    "default": DEFAULT_TIMEOUT_MINUTES, "min": 1, "max": 240,
                    "tooltip": "Wie lange die Node auf das Ergebnis wartet. Läuft die "
                               "Zeit ab, bricht nur die Node ab — der Job läuft "
                               "serverseitig weiter (und kostet trotzdem Credits)."}),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Leer = FAL_KEY aus .env oder Umgebungsvariable."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "metadata")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    async def generate(self, prompt, image_size, width, height,
                       num_inference_steps, guidance_scale, num_images, seed,
                       output_format, negative_prompt="",
                       enable_prompt_expansion=False,
                       enable_agentic_generation=False,
                       agentic_max_iterations=2,
                       agentic_samples_per_iteration=2,
                       agentic_early_stop=True,
                       enable_safety_checker=True,
                       sync_mode=False,
                       timeout_minutes=DEFAULT_TIMEOUT_MINUTES,
                       api_key=""):
        if not prompt.strip():
            raise ValueError("Cosmos3: prompt darf nicht leer sein.")
        if image_size not in IMAGE_SIZES:
            raise ValueError(
                f"Cosmos3: image_size '{image_size}' nicht unterstützt. "
                f"Erlaubt: {', '.join(IMAGE_SIZES)}."
            )

        payload: dict = {
            "prompt": prompt.strip(),
            "image_size": ({"width": int(width), "height": int(height)}
                           if image_size == "custom" else image_size),
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "num_images": int(num_images),
            "seed": int(seed),
            "output_format": output_format,
            "enable_safety_checker": bool(enable_safety_checker),
            "sync_mode": bool(sync_mode),
        }

        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt.strip()

        if enable_prompt_expansion:
            payload["enable_prompt_expansion"] = True

        if enable_agentic_generation:
            payload["enable_agentic_generation"] = True
            payload["agentic_max_iterations"] = int(agentic_max_iterations)
            payload["agentic_samples_per_iteration"] = int(agentic_samples_per_iteration)
            payload["agentic_early_stop"] = bool(agentic_early_stop)

        client = FalClient(api_key)
        task = await asyncio.to_thread(client.submit, payload, ENDPOINT_T2I)
        result = await client.poll_async(task, timeout=timeout_minutes * 60)

        images = result.get("images") or []
        if not images:
            raise RuntimeError(f"Cosmos3: Request fertig, aber ohne Bilder: {result}")

        frames = []
        for entry in images:
            url = entry.get("url", "")
            if not url.startswith("http"):
                raise RuntimeError(f"Cosmos3: ungültige Bild-URL im Ergebnis: {entry}")
            data = await asyncio.to_thread(client.download, url)
            frames.append(await asyncio.to_thread(bytes_to_image_tensor, data))

        batch = torch.cat(frames, dim=0)
        used_seed = int(result.get("seed", seed))
        return (batch, used_seed, format_metadata(payload, result, task, ENDPOINT_T2I))


NODE_CLASS_MAPPINGS = {
    "Cosmos3SuperTextToImage_DRE": Cosmos3SuperTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Cosmos3SuperTextToImage_DRE": "🌌 Cosmos 3 Super T2I (fal.ai) *DRE",
}
