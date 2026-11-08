"""
Prepare the 1,500-Sample High-End Photography & Matting Dataset for training
`flux3_ultra_rgba_vae.safetensors` (combining UltraFlux arXiv:2511.18050 Non-Adversarial
4K Wavelet Post-Training + 4-Channel AlphaVAE).

Uses direct HTTP download + local `pyarrow.parquet.ParquetFile` (avoiding remote Arrow stream crashes):
  1. 800 Base Multi-Style RGBA Layers (`datasets/flux3_transparent/images/*.png`)
  2. 250 Specialized RGBA Layers (`furry`, `ink`, `line_draw`, `neon_graffiti`) from `artplus/PrismLayersPro`
  3. 250 High-End Photography Native 1:1 1024x1024 Crops:
     - `tonyassi/vogue-runway-top15` (Vogue Runway editorial fashion photography)
     - `mAiello00/DIV2K` (`DIV2K_train_HR/*.png` uncompressed 2K DSLR reference benchmark photos)
  4. 200 High-Resolution Fine-Alpha & Translucent Multi-Layer Matting Composites combining
     Vogue/DIV2K high-frequency photography with fine fur/hair/ink/lace alpha masks.
"""

import io
import os
import json
import glob
import random
import requests
import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSPARENT_IMG_DIR = os.path.join(BASE_DIR, "datasets", "flux3_transparent", "images")
PRISM_CACHE_DIR = os.path.join(BASE_DIR, "datasets", "prism_cache")
OUT_DIR = os.path.join(BASE_DIR, "datasets", "flux3_ultra_vae")
IMG_DIR = os.path.join(OUT_DIR, "images")


def compute_high_freq_score(rgb_arr: np.ndarray) -> float:
    """Fast Laplacian/gradient energy score to pick the sharpest 1:1 1024x1024 patch from a 2K photo."""
    gray = rgb_arr.mean(axis=2).astype(np.float32)
    gx = np.abs(gray[:, 2:] - gray[:, :-2]).mean()
    gy = np.abs(gray[2:, :] - gray[:-2, :]).mean()
    return float(gx + gy)


def extract_best_native_1024_crop(im: Image.Image, rng: random.Random, tries: int = 5) -> Image.Image:
    """Extracts a 1024x1024 1:1 native pixel crop (no downscaling) with highest high-frequency detail."""
    w, h = im.size
    if w < 1024 or h < 1024:
        s = max(1024 / w, 1024 / h)
        im = im.resize((int(round(w * s)), int(round(h * s))), Image.Resampling.LANCZOS)
        w, h = im.size

    best_crop = None
    best_score = -1.0
    for _ in range(tries):
        x0 = rng.randint(0, w - 1024)
        y0 = rng.randint(0, h - 1024)
        crop = im.crop((x0, y0, x0 + 1024, y0 + 1024))
        arr = np.array(crop)[::4, ::4, :3]
        score = compute_high_freq_score(arr)
        if score > best_score:
            best_score = score
            best_crop = crop
    return best_crop


def fit_rgba_to_square(pil_img: Image.Image, resolution: int = 1024) -> Image.Image:
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
    return canvas


def download_file(url: str, dest: str, min_bytes: int = 10_000) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes:
        return dest
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def extract_prism_shard_layers(shard_rel: str, tag: str, target_count: int) -> list:
    existing = sorted(glob.glob(os.path.join(IMG_DIR, f"prism_{tag}_*.png")))
    if len(existing) >= target_count:
        print(f"  -> Reused {target_count} cached layers for '{tag}'")
        return existing[:target_count]

    local_parquet = os.path.join(PRISM_CACHE_DIR, os.path.basename(shard_rel))
    url = f"https://huggingface.co/datasets/artplus/PrismLayersPro/resolve/main/{shard_rel}"
    print(f"  -> Downloading PrismLayersPro shard {os.path.basename(shard_rel)} ...")
    download_file(url, local_parquet, min_bytes=10_000_000)

    pf = pq.ParquetFile(local_parquet)
    max_layers = 8
    cols = ["id", "layer_count"] + [f"layer_{i:02d}" for i in range(max_layers)]
    avail_cols = [c for c in cols if c in pf.schema_arrow.names]

    saved_paths = []
    for batch in pf.iter_batches(batch_size=16, columns=avail_cols):
        if len(saved_paths) >= target_count:
            break
        d = batch.to_pydict()
        n_rows = len(d["id"])
        for r in range(n_rows):
            if len(saved_paths) >= target_count:
                break
            lc = min(int(d["layer_count"][r] or 0), max_layers)
            for li in range(lc):
                if len(saved_paths) >= target_count:
                    break
                col = f"layer_{li:02d}"
                if col not in d or d[col][r] is None:
                    continue
                cell = d[col][r]
                raw_bytes = cell.get("bytes") if isinstance(cell, dict) else None
                if not raw_bytes:
                    continue
                try:
                    im = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
                    a_arr = np.array(im.getchannel("A"))
                    fg_ratio = float((a_arr > 15).mean())
                    if fg_ratio < 0.04 or fg_ratio > 0.85:
                        continue
                    sq = fit_rgba_to_square(im, resolution=1024)
                    out_p = os.path.join(IMG_DIR, f"prism_{tag}_{len(saved_paths):04d}.png")
                    sq.save(out_p, compress_level=1)
                    saved_paths.append(out_p)
                except Exception:
                    continue
    print(f"  -> Extracted {len(saved_paths)}/{target_count} high-res RGBA layers for '{tag}'")
    return saved_paths


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(PRISM_CACHE_DIR, exist_ok=True)
    rng = random.Random(20261108)
    manifest = []

    # -------------------------------------------------------------------------
    # 1. Existing 800 Multi-Style RGBA Layers (`datasets/flux3_transparent/images/*.png`)
    # -------------------------------------------------------------------------
    base_rgba_files = sorted(glob.glob(os.path.join(TRANSPARENT_IMG_DIR, "*.png")))
    print(f"[1/4] Registering {len(base_rgba_files)} existing RGBA layers from {TRANSPARENT_IMG_DIR} ...")
    for p in base_rgba_files:
        manifest.append({"path": p, "type": "rgba_layer", "source": "PrismLayersPro_Base800"})

    # -------------------------------------------------------------------------
    # 2. Extract 250 Specialized RGBA Layers (furry, ink, line_draw, neon_graffiti)
    # -------------------------------------------------------------------------
    print("[2/4] Extracting 250 specialized RGBA layers (furry, ink, line_draw, neon_graffiti) ...")
    extra_specs = [
        ("data/furry-00000-of-00013.parquet", "furry", 75),
        ("data/ink-00000-of-00011.parquet", "ink", 65),
        ("data/line_draw-00000-of-00010.parquet", "line_draw", 55),
        ("data/neon_graffiti-00000-of-00006.parquet", "neon_graffiti", 55),
    ]
    special_rgba_files = []
    for shard_rel, tag, count in extra_specs:
        paths = extract_prism_shard_layers(shard_rel, tag, count)
        special_rgba_files.extend(paths)
        for p in paths:
            manifest.append({"path": p, "type": "rgba_layer", "source": f"PrismLayersPro_{tag}"})

    # -------------------------------------------------------------------------
    # 3. Download 250 High-End Photography Native 1:1 Crops (Vogue Runway + DIV2K 2K DSLR)
    # -------------------------------------------------------------------------
    print("[3/4] Downloading High-End Photography (Vogue Runway + DIV2K 2K DSLR) ...")
    high_end_photos = []

    # 3A: Vogue Runway (`tonyassi/vogue-runway-top15`) via parquet shards
    vogue_existing = sorted(glob.glob(os.path.join(IMG_DIR, "hf_vogue_*.png")))
    if len(vogue_existing) >= 130:
        high_end_photos.extend(vogue_existing[:130])
        print("  -> Reused 130 cached Vogue Runway high-end photos.")
    else:
        vogue_saved = 0
        for s_idx in range(6):
            if vogue_saved >= 130:
                break
            shard_name = f"train-{s_idx:05d}-of-00215.parquet"
            v_url = f"https://huggingface.co/datasets/tonyassi/vogue-runway-top15/resolve/main/data/{shard_name}"
            v_dest = os.path.join(PRISM_CACHE_DIR, f"vogue_{shard_name}")
            try:
                download_file(v_url, v_dest, min_bytes=1_000_000)
                pf = pq.ParquetFile(v_dest)
                for batch in pf.iter_batches(batch_size=16):
                    if vogue_saved >= 130:
                        break
                    d = batch.to_pydict()
                    for cell in d.get("image", []):
                        if vogue_saved >= 130:
                            break
                        raw = cell.get("bytes") if isinstance(cell, dict) else None
                        if not raw:
                            continue
                        im = Image.open(io.BytesIO(raw)).convert("RGB")
                        crop = extract_best_native_1024_crop(im, rng, tries=5)
                        out_p = os.path.join(IMG_DIR, f"hf_vogue_{vogue_saved:04d}.png")
                        crop.convert("RGBA").save(out_p, compress_level=1)
                        high_end_photos.append(out_p)
                        vogue_saved += 1
            except Exception as e:
                print(f"  [Warning] Vogue shard {s_idx}: {e}")
        print(f"  -> Saved {vogue_saved} native 1:1 1024x1024 crops from Vogue Runway.")

    # 3B: DIV2K Uncompressed 2K DSLR Reference Photos (`mAiello00/DIV2K`)
    div2k_existing = sorted(glob.glob(os.path.join(IMG_DIR, "hf_div2k_*.png")))
    if len(div2k_existing) >= 120:
        high_end_photos.extend(div2k_existing[:120])
        print("  -> Reused 120 cached DIV2K 2K DSLR reference photos.")
    else:
        div2k_saved = 0
        for img_idx in range(1, 160):
            if div2k_saved >= 120:
                break
            out_p = os.path.join(IMG_DIR, f"hf_div2k_{div2k_saved:04d}.png")
            if os.path.exists(out_p):
                high_end_photos.append(out_p)
                div2k_saved += 1
                continue
            d_url = f"https://huggingface.co/datasets/mAiello00/DIV2K/resolve/main/DIV2K_train_HR/{img_idx:04d}.png"
            try:
                r = requests.get(d_url, timeout=60)
                if r.status_code != 200:
                    continue
                im = Image.open(io.BytesIO(r.content)).convert("RGB")
                crop = extract_best_native_1024_crop(im, rng, tries=6)
                crop.convert("RGBA").save(out_p, compress_level=1)
                high_end_photos.append(out_p)
                div2k_saved += 1
                if div2k_saved % 25 == 0:
                    print(f"    DIV2K 2K DSLR progress: {div2k_saved}/120")
            except Exception:
                continue
        print(f"  -> Saved {div2k_saved} native 1:1 1024x1024 crops from DIV2K 2K DSLR.")

    for p in high_end_photos:
        manifest.append({"path": p, "type": "opaque_highres_rgb", "source": "HF_Vogue_DIV2K_2K"})

    # -------------------------------------------------------------------------
    # 4. Build 200 High-Res Fine-Alpha & Translucent Matting Composites
    #    (pairing Vogue/DIV2K 2K DSLR textures with Furry/Ink/Line-Art/Silk alpha masks)
    # -------------------------------------------------------------------------
    print("[4/4] Building 200 High-Res Fine-Alpha & Translucent Matting samples ...")
    mask_sources = special_rgba_files + base_rgba_files
    for i in range(200):
        out_p = os.path.join(IMG_DIR, f"hf_fine_alpha_matting_{i:04d}.png")
        if not os.path.exists(out_p):
            photo_p = high_end_photos[i % len(high_end_photos)]
            mask_p = mask_sources[i % len(mask_sources)]
            with Image.open(photo_p) as pim:
                rgb_arr = np.array(pim.convert("RGB"), dtype=np.uint8)
            with Image.open(mask_p) as mim:
                mim_1k = mim.convert("RGBA").resize((1024, 1024), Image.Resampling.LANCZOS)
                a_arr = np.array(mim_1k.split()[3], dtype=np.float32) / 255.0

            # For half the samples, add smooth translucent interior gradients (silk/glass/smoke)
            if i % 2 == 1:
                yy, xx = np.mgrid[0:1024, 0:1024].astype(np.float32) / 1024.0
                wave = 0.55 + 0.45 * np.sin(xx * 3.5 * np.pi + yy * 2.5 * np.pi)
                a_arr = np.where(a_arr > 0.02, np.clip(a_arr * wave, 0.0, 1.0), 0.0)

            alpha_u8 = (a_arr * 255.0).clip(0, 255).astype(np.uint8)
            rgba_arr = np.dstack([rgb_arr, alpha_u8])
            Image.fromarray(rgba_arr, mode="RGBA").save(out_p, compress_level=1)
        manifest.append({"path": out_p, "type": "fine_alpha_highres", "source": "HF_FineAlpha_2K_Matting"})

    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 72)
    print(f"[UltraRGBAVAE Dataset] Ready! Total samples: {len(manifest)}")
    print(f"Manifest saved to: {manifest_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
