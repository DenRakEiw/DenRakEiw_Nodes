#!/usr/bin/env python3
"""
Dataset Creator for FLUX 3 Image Klein Transparent LoRA (Text-to-Transparent-Image).
Extracts curated RGBA object layers + captions from PrismLayersPro, applies pyramid
alpha-bleeding (pad_rgb), computes pixel-accurate edge-loss weight maps, and encodes
128-channel latents using the Flux 2/3 RGBA VAE (trmz/flux2-klein-alpha).
"""

import argparse
import io
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
import pyarrow.parquet as pq

# Add ComfyUI root and local dir to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from flux_rgba_vae_nodes import (
    load_flux23_rgba_vae,
    HF_RGBA_VAE_URL,
    DEFAULT_RGBA_VAE_FILENAME,
)
from transparent_vae import pad_rgb

PRISM_SHARDS = [
    "data/3D-00001-of-00002.parquet",
    "data/toy-00000-of-00015.parquet",
    "data/watercolor_painting-00000-of-00010.parquet",
    "data/cartoon-00000-of-00009.parquet",
    "data/anime-00000-of-00007.parquet",
    "data/steampunk-00000-of-00012.parquet",
]


def compute_edge_loss_weight(
    alpha_hw: np.ndarray,
    latent_h: int,
    latent_w: int,
    edge_multiplier: float = 3.0,
    dilate: int = 1,
) -> torch.Tensor:
    """Compute the mean-1.0 spatial edge loss weight map at latent resolution
    from the high-res alpha channel (following Xavier Jara / ai-toolkit-rgba §7)."""
    ch = torch.from_numpy(alpha_hw).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    gx = F.pad((ch[:, :, :, 1:] - ch[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((ch[:, :, 1:, :] - ch[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    edge = gx + gy

    # Max-pool to latent resolution so 1px edges survive
    edge = F.adaptive_max_pool2d(edge, (latent_h, latent_w))
    if dilate > 0:
        edge = F.max_pool2d(edge, kernel_size=dilate * 2 + 1, stride=1, padding=dilate)

    peak = edge.amax(dim=(2, 3), keepdim=True).clamp(min=1e-4)
    edge = (edge / peak).clamp(0.0, 1.0)

    weight = 1.0 + edge_multiplier * edge
    weight = weight / weight.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
    return weight.squeeze(0)  # [1, latent_h, latent_w]


def fit_rgba_to_square(pil_img: Image.Image, resolution: int = 512) -> np.ndarray:
    """Fit an RGBA PIL image onto a transparent square canvas of `resolution x resolution`
    with a small margin, returning uint8 [H, W, 4]."""
    pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)
    alpha = arr[..., 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) > 10 and len(ys) > 10:
        x0, x1 = max(0, int(xs.min()) - 4), min(arr.shape[1], int(xs.max()) + 5)
        y0, y1 = max(0, int(ys.min()) - 4), min(arr.shape[0], int(ys.max()) + 5)
        pil_img = pil_img.crop((x0, y0, x1, y1))

    w, h = pil_img.size
    target_inner = int(resolution * 0.90)
    scale = min(target_inner / max(w, 1), target_inner / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    offset_x = (resolution - new_w) // 2
    offset_y = (resolution - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.array(canvas, dtype=np.uint8)


def ensure_rgba_vae(comfy_root: str) -> str:
    vae_dir = os.path.join(comfy_root, "models", "vae")
    os.makedirs(vae_dir, exist_ok=True)
    vae_path = os.path.join(vae_dir, DEFAULT_RGBA_VAE_FILENAME)
    if not os.path.exists(vae_path) or os.path.getsize(vae_path) < 100_000_000:
        print(f"Downloading {DEFAULT_RGBA_VAE_FILENAME} from HuggingFace...")
        with requests.get(HF_RGBA_VAE_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(vae_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
    return vae_path


def download_shard(shard_rel: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    local_name = os.path.basename(shard_rel)
    dest = os.path.join(cache_dir, local_name)
    if os.path.exists(dest) and os.path.getsize(dest) > 50_000_000:
        return dest
    url = f"https://huggingface.co/datasets/artplus/PrismLayersPro/resolve/main/{shard_rel}"
    print(f"Downloading dataset shard {local_name} from HuggingFace...")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def clean_caption(raw_caption: str, trigger_word: str = "transparent background") -> str:
    cap = (raw_caption or "").strip()
    for prefix in [
        "The image features ",
        "The picture shows ",
        "The picture features ",
        "The image shows ",
        "This image features ",
    ]:
        if cap.startswith(prefix):
            cap = cap[len(prefix):]
            if cap:
                cap = cap[0].upper() + cap[1:]
            break
    if trigger_word and trigger_word.lower() not in cap.lower():
        cap = f"{cap.rstrip('.')}, {trigger_word}."
    return cap


def build_dataset(
    output_dir: str = "datasets/flux3_transparent",
    target_count: int = 600,
    resolution: int = 512,
    trigger_word: str = "transparent background",
):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    cache_dir = os.path.join(output_dir, "cache")
    prism_cache = os.path.join("datasets", "prism_cache")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    vae_path = ensure_rgba_vae(COMFY_ROOT)
    print(f"Loading RGBA VAE from: {vae_path}")
    rgba_vae = load_flux23_rgba_vae(vae_path, dtype_str="bfloat16")

    max_layers_Check = 12
    cols = (
        ["id", "style_category", "layer_count"]
        + [f"layer_{i:02d}" for i in range(max_layers_Check)]
        + [f"layer_{i:02d}_caption" for i in range(max_layers_Check)]
    )

    saved = 0
    manifest = []
    per_shard_limit = max(1, (target_count + len(PRISM_SHARDS) - 1) // len(PRISM_SHARDS))

    for shard_rel in PRISM_SHARDS:
        if saved >= target_count:
            break
        shard_path = download_shard(shard_rel, prism_cache)
        pf = pq.ParquetFile(shard_path)
        avail_cols = [c for c in cols if c in pf.schema_arrow.names]
        shard_saved = 0

        for batch in pf.iter_batches(batch_size=16, columns=avail_cols):
            if saved >= target_count or shard_saved >= per_shard_limit:
                break
            d = batch.to_pydict()
            n_rows = len(d["id"])
            for r in range(n_rows):
                if saved >= target_count or shard_saved >= per_shard_limit:
                    break
                lc = min(int(d["layer_count"][r] or 0), max_layers_Check)
                row_id = d["id"][r]
                style = d.get("style_category", [""] * n_rows)[r] or ""

                # Iterate layers (skip layer_00 if opaque background)
                for li in range(lc):
                    if saved >= target_count or shard_saved >= per_shard_limit:
                        break
                    k = f"layer_{li:02d}"
                    if k not in d:
                        continue
                    ld = d[k][r]
                    raw_cap = str(d.get(f"{k}_caption", [""] * n_rows)[r] or "").strip()
                    if not ld or not ld.get("bytes") or not raw_cap:
                        continue
                    # Skip pure text overlays
                    if raw_cap.startswith("Text ") or raw_cap.startswith('Text "') or len(raw_cap) < 15:
                        continue

                    try:
                        im = Image.open(io.BytesIO(ld["bytes"])).convert("RGBA")
                    except Exception:
                        continue

                    w, h = im.size
                    if min(w, h) < 180:
                        continue

                    arr = np.array(im)
                    alpha = arr[..., 3]
                    a_min, a_max, a_mean = int(alpha.min()), int(alpha.max()), float(alpha.mean())
                    # Require genuine transparency (background=0, foreground=255, reasonable coverage)
                    transparent_frac = float((alpha < 10).mean())
                    opaque_frac = float((alpha > 240).mean())
                    if a_min > 5 or a_max < 245 or transparent_frac < 0.15 or opaque_frac < 0.05 or opaque_frac > 0.85:
                        continue

                    # Fit onto square RGBA canvas
                    square_rgba_u8 = fit_rgba_to_square(im, resolution=resolution)
                    square_rgba_f32 = square_rgba_u8.astype(np.float32) / 255.0

                    # Premultiply RGB by alpha so transparent background is clean (0, 0, 0, 0),
                    # matching trmz/flux2-klein-alpha's Premultiplied RGB VAE training
                    alpha_ch = square_rgba_f32[..., 3:4]
                    premul_rgb = np.clip(square_rgba_f32[..., :3] * alpha_ch, 0.0, 1.0)
                    premul_rgba = np.concatenate([premul_rgb, alpha_ch], axis=-1)

                    # Encode with Flux 2/3 RGBA VAE -> [1, 128, H/16, W/16]
                    rgba_tensor = torch.from_numpy(premul_rgba).unsqueeze(0)  # [1, H, W, 4]
                    with torch.no_grad():
                        latent = rgba_vae.encode(rgba_tensor)[0].to(torch.bfloat16).cpu()  # [128, H/16, W/16]

                    latent_h, latent_w = latent.shape[1], latent.shape[2]
                    edge_weight = compute_edge_loss_weight(
                        square_rgba_f32[..., 3],
                        latent_h=latent_h,
                        latent_w=latent_w,
                        edge_multiplier=3.0,
                        dilate=1,
                    )

                    caption = clean_caption(raw_cap, trigger_word=trigger_word)
                    item_name = f"sample_{saved:05d}_{row_id}_L{li:02d}"

                    # Save visual PNG (original unbled RGBA for preview/inspection) + caption .txt
                    png_path = os.path.join(images_dir, f"{item_name}.png")
                    txt_path = os.path.join(images_dir, f"{item_name}.txt")
                    Image.fromarray(square_rgba_u8, mode="RGBA").save(png_path, "PNG")
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(caption)

                    # Save pre-encoded training sample
                    pt_path = os.path.join(cache_dir, f"{item_name}.pt")
                    torch.save(
                        {
                            "latent": latent,
                            "edge_weight": edge_weight,
                            "caption": caption,
                            "style": style,
                            "resolution": resolution,
                        },
                        pt_path,
                    )

                    manifest.append(
                        {
                            "id": item_name,
                            "caption": caption,
                            "style": style,
                            "png": png_path,
                            "cache": pt_path,
                        }
                    )
                    saved += 1
                    shard_saved += 1
                    if saved % 50 == 0 or saved == target_count:
                        print(f"  [{saved}/{target_count}] ({style}) Prepared & encoded {item_name}: {caption[:60]}...")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)

    print(f"\n✓ Dataset creation complete! Saved {saved} RGBA samples to {output_dir}")
    print(f"  - PNGs + Captions: {images_dir}")
    print(f"  - Latents + Edge Maps: {cache_dir}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FLUX 3 Klein Transparent RGBA Dataset")
    parser.add_argument("--output_dir", type=str, default="datasets/flux3_transparent")
    parser.add_argument("--count", type=int, default=500, help="Number of transparent RGBA layers to prepare")
    parser.add_argument("--resolution", type=int, default=512, help="Square training resolution (e.g. 512)")
    parser.add_argument("--trigger", type=str, default="transparent background")
    args = parser.parse_args()

    build_dataset(
        output_dir=args.output_dir,
        target_count=args.count,
        resolution=args.resolution,
        trigger_word=args.trigger,
    )
