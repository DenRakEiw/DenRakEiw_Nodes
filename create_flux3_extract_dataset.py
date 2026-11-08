"""
Build the 800-sample Foreground-Extraction dataset (`datasets/flux3_extract/`) for FLUX 3 Image Klein.

For each of the 800 clean RGBA foreground samples in `datasets/flux3_transparent/`:
  1. Composites the RGBA foreground onto a realistic background (real `base_image` scenes from
     `PrismLayersPro` parquets + diverse procedural studio/bokeh/gradient/textured backgrounds).
  2. Saves the composite RGB image (`datasets/flux3_extract/composites/sample_XXXXX.png`).
  3. Encodes the composite image (opaque alpha=1.0) through `flux2_3_rgba_vae.safetensors` into
     `ref_latent` ([128, 32, 32]) and pairs it with the clean Premultiplied RGBA `target_latent`
     ([128, 32, 32]) and `edge_weight` ([1, 32, 32]).
"""

import os
import io
import glob
import random
import numpy as np
import torch
import pyarrow.parquet as pq
from PIL import Image, ImageFilter, ImageDraw

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

from flux_rgba_vae_nodes import load_flux23_rgba_vae, DEFAULT_RGBA_VAE_FILENAME


def load_real_backgrounds(parquet_dir: str, max_bgs: int = 400, size: int = 512) -> list[np.ndarray]:
    """Extract real background plates (`base_image`) from cached PrismLayersPro parquet files."""
    pfiles = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    bgs = []
    for pf_path in pfiles:
        try:
            pf = pq.ParquetFile(pf_path)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["base_image"])
                col = table.column("base_image").to_pylist()
                for item in col:
                    if item and isinstance(item, dict) and item.get("bytes"):
                        try:
                            im = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
                            im = im.resize((size, size), Image.Resampling.LANCZOS)
                            arr = np.asarray(im, dtype=np.float32) / 255.0
                            if arr.std() > 0.05:
                                bgs.append(arr)
                                if len(bgs) >= max_bgs:
                                    return bgs
                        except Exception:
                            pass
        except Exception as e:
            print(f"Warning reading {pf_path}: {e}")
    return bgs


def make_procedural_background(size: int = 512, rng: random.Random = None) -> np.ndarray:
    """Create a rich synthetic background (gradients, bokeh circles, geometric distractors, noise)."""
    if rng is None:
        rng = random
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)

    # Random linear + radial gradient
    c1 = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
    c2 = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
    c3 = np.array([rng.random(), rng.random(), rng.random()], dtype=np.float32)
    angle = rng.uniform(0, 6.28318)
    t = np.clip((xx - 0.5) * np.cos(angle) + (yy - 0.5) * np.sin(angle) + 0.5, 0.0, 1.0)
    bg = c1[None, None, :] * (1.0 - t[..., None]) + c2[None, None, :] * t[..., None]

    cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    rad = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * rng.uniform(0.18, 0.45) ** 2))
    bg = bg * (1.0 - 0.5 * rad[..., None]) + c3[None, None, :] * (0.5 * rad[..., None])

    # Add bokeh / distractor shapes via PIL
    pil_bg = Image.fromarray((bg.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil_bg, "RGBA")
    num_shapes = rng.randint(6, 18)
    for _ in range(num_shapes):
        x0 = rng.randint(-60, size - 20)
        y0 = rng.randint(-60, size - 20)
        w = rng.randint(40, 220)
        h = rng.randint(40, 220)
        col = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255), rng.randint(40, 160))
        if rng.random() < 0.65:
            draw.ellipse([x0, y0, x0 + w, y0 + h], fill=col)
        else:
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=col)

    blur_r = rng.choice([0, 2, 6, 12])
    if blur_r > 0:
        pil_bg = pil_bg.filter(ImageFilter.GaussianBlur(blur_r))

    return np.asarray(pil_bg, dtype=np.float32) / 255.0


def build_extract_dataset(
    src_dir: str = "datasets/flux3_transparent",
    dst_dir: str = "datasets/flux3_extract",
    parquet_dir: str = "datasets/prism_cache",
    size: int = 512,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_cache_files = sorted(glob.glob(os.path.join(src_dir, "cache", "*.pt")))
    if not src_cache_files:
        raise FileNotFoundError(f"No source cache files in {src_dir}/cache")

    comp_dir = os.path.join(dst_dir, "composites")
    cache_dir = os.path.join(dst_dir, "cache")
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading real background plates from {parquet_dir}...")
    real_bgs = load_real_backgrounds(parquet_dir, max_bgs=400, size=size)
    print(f"[OK] Loaded {len(real_bgs)} real background scenes.")

    vae_path = os.path.join(COMFY_ROOT, "models", "vae", DEFAULT_RGBA_VAE_FILENAME)
    print(f"Loading 4-channel RGBA VAE from {vae_path}...")
    vae = load_flux23_rgba_vae(vae_path, dtype_str="bfloat16")

    rng = random.Random(42)
    for i, src_pt in enumerate(src_cache_files):
        base_name = os.path.splitext(os.path.basename(src_pt))[0]
        dst_pt = os.path.join(cache_dir, f"{base_name}.pt")
        comp_png = os.path.join(comp_dir, f"{base_name}.png")

        if os.path.exists(dst_pt) and os.path.exists(comp_png):
            continue

        src_data = torch.load(src_pt, map_location="cpu")
        fg_png = os.path.join(src_dir, "images", f"{base_name}.png")
        fg_im = Image.open(fg_png).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
        fg_np = np.asarray(fg_im, dtype=np.float32) / 255.0

        fg_rgb = fg_np[..., :3]
        fg_a = fg_np[..., 3:4]

        # Pick real background 55% of the time, procedural background 45% of the time
        if real_bgs and rng.random() < 0.55:
            bg_rgb = real_bgs[rng.randrange(len(real_bgs))].copy()
            # Random flip/color jitter for variety
            if rng.random() < 0.5:
                bg_rgb = bg_rgb[:, ::-1, :]
        else:
            bg_rgb = make_procedural_background(size=size, rng=rng)

        # Composite foreground over background
        comp_rgb = (fg_rgb * fg_a + bg_rgb * (1.0 - fg_a)).clip(0.0, 1.0)
        Image.fromarray((comp_rgb * 255.0).astype(np.uint8), mode="RGB").save(comp_png)

        # Encode opaque composite [1, H, W, 4] (with alpha=1.0) through RGBA VAE
        comp_rgba = np.concatenate([comp_rgb, np.ones_like(fg_a)], axis=-1)
        comp_t = torch.from_numpy(comp_rgba).unsqueeze(0)
        with torch.no_grad():
            ref_latent = vae.encode(comp_t)[0].to(torch.bfloat16).cpu()

        # Construct extraction caption from original caption
        orig_cap = src_data.get("caption", "")
        # Strip old prefix if present
        clean_desc = orig_cap.replace("RGBA transparent background, isolated foreground object,", "").strip()
        extract_caption = f"Extract foreground object, RGBA transparent background: {clean_desc}"

        out_data = {
            "target_latent": src_data["latent"].to(torch.bfloat16).cpu(),   # Clean Premultiplied RGBA [128, 32, 32]
            "ref_latent": ref_latent,                                        # Opaque Composite [128, 32, 32]
            "edge_weight": src_data["edge_weight"].to(torch.float32).cpu(), # [1, 32, 32]
            "caption": extract_caption,
        }
        torch.save(out_data, dst_pt)

        if (i + 1) % 100 == 0 or (i + 1) == len(src_cache_files):
            print(f"  [{i + 1}/{len(src_cache_files)}] Built composite & encoded ref_latent -> {base_name}")

    print(f"[OK] Built {len(src_cache_files)} Foreground-Extraction samples in {dst_dir}!")


if __name__ == "__main__":
    build_extract_dataset()
