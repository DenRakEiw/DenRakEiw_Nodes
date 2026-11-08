"""
Standalone PyTorch LoRA Trainer for FLUX 3 Image Klein (5.6B) — Native RGBA Foreground Extraction
=================================================================================================
Trains a Rank-32 LoRA (`flux3_klein_extract_lora.safetensors`) on `flux3-image-klein-base-bf16.safetensors`
that takes a composite reference image (`ReferenceLatent` / `image_cond`) + prompt and extracts the
foreground subject into a 4-channel Premultiplied RGBA latent for `flux2_3_rgba_vae.safetensors`.
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import gc
import glob
import math
import time
import random
import argparse
import requests
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

import comfy.utils
import comfy.ops
import comfy.sd
import comfy.model_management
from comfy.ldm.flux3.model import Flux3
from comfy.ldm.flux.layers import timestep_embedding
from flux_rgba_vae_nodes import DEFAULT_RGBA_VAE_FILENAME
from train_flux3_transparent_lora import (
    FLUX3_SHIFT,
    LoRALinear,
    trainable_block_forward,
    ensure_premul_latents_and_alpha_weights,
    sample_flux3_timestep,
    compute_sharpness_loss,
    save_comfy_lora,
)

GENERIC_EXTRACT_PROMPT = (
    "Extract the main foreground subject, RGBA transparent background, isolated foreground object"
)


def inject_extract_lora(
    model: Flux3,
    rank: int = 32,
    alpha: float = 32.0,
    device: torch.device = torch.device("cuda"),
    init_lora_path: str = None,
) -> dict[str, LoRALinear]:
    """
    Inject LoRA adapters into `image`, `single_stream_modulations.image_cond`, and `single_blocks` (204 layers).
    Warm-starts matching layers from `flux3_klein_transparent_lora.safetensors`.
    """
    target_attr_names = {"q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out", "lin"}
    lora_modules: dict[str, LoRALinear] = {}

    for module_path, module in list(model.named_modules()):
        if not (
            module_path.startswith("content_mode_blocks.image.")
            or module_path.startswith("single_blocks.")
            or module_path.startswith("early_stream_modulations.image")
            or module_path.startswith("single_stream_modulations.")
            or module_path in ("emb_in.image", "final_layer.image.linear")
        ):
            continue

        parts = module_path.split(".")
        leaf_name = parts[-1]
        if leaf_name not in target_attr_names and module_path not in (
            "emb_in.image",
            "final_layer.image.linear",
        ):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, device=device)
        setattr(parent, leaf_name, lora_layer)
        lora_modules[module_path] = lora_layer

    if init_lora_path and os.path.exists(init_lora_path):
        init_sd = comfy.utils.load_torch_file(init_lora_path)
        loaded_count = 0
        with torch.no_grad():
            for path, mod in lora_modules.items():
                k_down = f"diffusion_model.{path}.lora_down.weight"
                k_up = f"diffusion_model.{path}.lora_up.weight"
                if k_down in init_sd and k_up in init_sd:
                    if init_sd[k_down].shape == mod.lora_down.weight.shape:
                        mod.lora_down.weight.copy_(init_sd[k_down].to(device=device, dtype=torch.float32))
                        mod.lora_up.weight.copy_(init_sd[k_up].to(device=device, dtype=torch.float32))
                        loaded_count += 1
        print(f"[OK] Warm-started {loaded_count}/{len(lora_modules)} LoRA modules from {os.path.basename(init_lora_path)}")

    return lora_modules


def forward_flux3_extract_trainable(
    model: Flux3,
    x: torch.Tensor,
    timestep: torch.Tensor,
    ref_states: torch.Tensor,
    ref_shape: tuple[int, int],
    txt_states: torch.Tensor,
) -> torch.Tensor:
    """
    Fast, autograd-safe forward pass for FLUX 3 Image Klein using precomputed t=0 `ref_states` and `txt_states`.
    """
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device

    vec = model.time_in(timestep_embedding(torch.cat([timestep, timestep.new_zeros(1)]).to(dtype), 256))
    vec_img, vec_zero = vec[:-1], vec[-1:]

    early_img = model.early_stream_modulations["image"](vec_img)[0]
    img = model.emb_in["image"](x.flatten(2).transpose(1, 2))
    parts = [
        (img, "image", model.image_ids(H, W, 0, device)),
        (ref_states, "image_cond", model.ref_ids([ref_shape], device)),
        (txt_states, "txt", model.text_ids(txt_states.shape[1], device)),
    ]
    pe = model.rope(torch.cat([ids for _, _, ids in parts]))

    spans, start = [], 0
    for tokens, mode, _ in parts:
        mod = model.single_stream_modulations[mode](vec_img if mode == "image" else vec_zero)[0]
        spans.append((start, start + tokens.shape[1], mod))
        start += tokens.shape[1]

    pe_img = pe[:, : H * W]
    for block in model.content_mode_blocks["image"]:
        img = checkpoint(trainable_block_forward, block, img, [(0, H * W, early_img)], pe_img, use_reentrant=False)

    joint = torch.cat([img, ref_states, txt_states], dim=1)
    for block in model.single_blocks:
        joint = checkpoint(trainable_block_forward, block, joint, spans, pe, use_reentrant=False)

    out = model.final_layer["image"](joint[:, : H * W], vec_img)
    return out.transpose(1, 2).reshape(B, model.in_channels, H, W)


def cache_extract_text_embeddings(dataset_dir: str, cache_files: list[str], clip_path: str) -> dict:
    """Encode unconditional '', generic extract prompt, and per-sample specific captions with Qwen3-VL-4B."""
    meta_cache_path = os.path.join(dataset_dir, "extract_text_meta.pt")
    missing = []
    for pf in cache_files:
        d = torch.load(pf, map_location="cpu")
        if "txt_states" not in d and "context" not in d:
            missing.append(pf)

    need_meta = not os.path.exists(meta_cache_path)
    if not missing and not need_meta:
        print(f"[OK] All {len(cache_files)} extract samples and global prompts already have cached embeddings.")
        return torch.load(meta_cache_path, map_location="cpu")

    print(f"Loading Qwen3-VL-4B (CLIPType.FLUX3) for text encoding (missing={len(missing)}, meta={need_meta})...")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.FLUX3)

    if need_meta:
        with torch.no_grad():
            t_uncond = clip.tokenize("")
            c_uncond = clip.encode_from_tokens_scheduled(t_uncond)[0][0].to(torch.bfloat16).cpu()

            t_gen = clip.tokenize(GENERIC_EXTRACT_PROMPT)
            c_gen = clip.encode_from_tokens_scheduled(t_gen)[0][0].to(torch.bfloat16).cpu()

            torch.save({"uncond_context": c_uncond, "generic_context": c_gen}, meta_cache_path)
            print(f"[OK] Encoded unconditional & generic extraction prompts.")

    for i, pf in enumerate(missing):
        d = torch.load(pf, map_location="cpu")
        caption = d["caption"]
        with torch.no_grad():
            tokens = clip.tokenize(caption)
            ctx = clip.encode_from_tokens_scheduled(tokens)[0][0].to(torch.bfloat16).cpu()
        d["context"] = ctx
        torch.save(d, pf)
        if (i + 1) % 100 == 0 or (i + 1) == len(missing):
            print(f"  [{i + 1}/{len(missing)}] Encoded specific caption -> {tuple(ctx.shape)}")

    del clip
    gc.collect()
    torch.cuda.empty_cache()
    return torch.load(meta_cache_path, map_location="cpu")


def train_extract(
    dataset_dir: str = "datasets/flux3_extract",
    transparent_dir: str = "datasets/flux3_transparent",
    steps: int = 600,
    lr: float = 1e-4,
    rank: int = 32,
    alpha: float = 32.0,
    grad_accum: int = 2,
    edge_multiplier: float = 3.0,
    sharpness_multiplier: float = 0.25,
    save_every: int = 200,
    output_name: str = "flux3_klein_extract_lora.safetensors",
):
    # Free idle ComfyUI VRAM before starting training so there is no shared-memory paging
    try:
        requests.post("http://127.0.0.1:8188/free", json={"unload_models": True, "free_memory": True}, timeout=3)
        print("[OK] Freed idle ComfyUI VRAM cache.")
    except Exception:
        pass

    comfy.model_management.in_training = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = os.path.join(dataset_dir, "cache")
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not cache_files:
        raise FileNotFoundError(f"No cached samples found in {cache_dir}. Run create_flux3_extract_dataset.py first!")

    clip_path = os.path.join(COMFY_ROOT, "models", "text_encoders", "f3_qwen3vl_4b.safetensors")
    model_path = os.path.join(COMFY_ROOT, "models", "diffusion_models", "flux3-image-klein-base-bf16.safetensors")
    loras_dir = os.path.join(COMFY_ROOT, "models", "loras")
    init_lora_path = os.path.join(loras_dir, "flux3_klein_transparent_lora.safetensors")

    # 1. Compute 128-ch Alpha sensitivity weights
    vae_path = os.path.join(COMFY_ROOT, "models", "vae", DEFAULT_RGBA_VAE_FILENAME)
    trans_cache_files = sorted(glob.glob(os.path.join(transparent_dir, "cache", "*.pt")))
    alpha_ch_weight = ensure_premul_latents_and_alpha_weights(
        transparent_dir, trans_cache_files, vae_path, alpha_channel_boost=3.0
    ).to(device=device, dtype=torch.float32)

    # 2. Encode captions
    meta_data = cache_extract_text_embeddings(dataset_dir, cache_files, clip_path)

    # 3. Load FLUX 3 Image Klein Base (5.6B) with _norm.scale -> _norm.weight fix
    print(f"Loading FLUX 3 Image Klein Base from: {model_path}")
    raw_sd = comfy.utils.load_torch_file(model_path)
    sd = {(k[:-6] + ".weight" if k.endswith("_norm.scale") else k): v for k, v in raw_sd.items()}
    del raw_sd
    model = Flux3(
        hidden_size=3072,
        num_heads=24,
        depth=5,
        depth_single_blocks=28,
        fused_qkv=False,
        flow_map=False,
        dtype=torch.bfloat16,
        device=device,
        operations=comfy.ops.disable_weight_init,
    )
    missing, unexpected = model.load_state_dict(sd, strict=True, assign=True)
    del sd
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[OK] Loaded FLUX 3 Base on {device} (missing={len(missing)}, unexpected={len(unexpected)})")

    # 4. Precompute t=0 txt_states AND ref_states (so training runs at maximum speed!)
    print(f"Preparing {len(cache_files)} extraction samples (computing t=0 txt_states & ref_states)...")
    with torch.no_grad():
        if "uncond_txt_states" in meta_data and "generic_txt_states" in meta_data:
            uncond_txt_states = meta_data["uncond_txt_states"].to(torch.bfloat16)
            generic_txt_states = meta_data["generic_txt_states"].to(torch.bfloat16)
        else:
            uctx = meta_data["uncond_context"].to(device=device, dtype=torch.bfloat16)
            uncond_txt_states, _ = model.preprocess_conds(uctx)
            uncond_txt_states = uncond_txt_states.detach().to(torch.bfloat16).cpu()

            gctx = meta_data["generic_context"].to(device=device, dtype=torch.bfloat16)
            generic_txt_states, _ = model.preprocess_conds(gctx)
            generic_txt_states = generic_txt_states.detach().to(torch.bfloat16).cpu()

            meta_data["uncond_txt_states"] = uncond_txt_states
            meta_data["generic_txt_states"] = generic_txt_states
            torch.save(meta_data, os.path.join(dataset_dir, "extract_text_meta.pt"))

        dummy_ctx = uncond_txt_states[:, :1, :].to(device=device, dtype=torch.bfloat16)
        vec_zero = model.time_in(timestep_embedding(dummy_ctx.new_zeros(1), 256))
        mod_ref = model.early_stream_modulations["image_cond"](vec_zero)[0]

        samples = []
        for i, pf in enumerate(cache_files):
            d = torch.load(pf, map_location="cpu")
            changed = False
            if "txt_states" in d:
                txt_states = d["txt_states"].to(torch.bfloat16)
            else:
                ctx = d["context"].to(device=device, dtype=torch.bfloat16)
                txt_states, _ = model.preprocess_conds(ctx)
                txt_states = txt_states.detach().to(torch.bfloat16).cpu()
                d["txt_states"] = txt_states
                if "context" in d:
                    del d["context"]
                changed = True

            ref_lat = d["ref_latent"]
            Hr, Wr = ref_lat.shape[-2], ref_lat.shape[-1]
            if "ref_states" in d:
                ref_states = d["ref_states"].to(torch.bfloat16)
            else:
                rl_gpu = ref_lat.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
                refs = model.emb_in["image_cond"](rl_gpu.flatten(2).transpose(1, 2))
                pe_ref = model.rope(model.ref_ids([(Hr, Wr)], device))
                for block in model.content_mode_blocks["image_cond"]:
                    refs = block(refs, [(0, Hr * Wr, mod_ref)], pe_ref)
                ref_states = refs.detach().to(torch.bfloat16).cpu()
                d["ref_states"] = ref_states
                changed = True

            if changed:
                torch.save(d, pf)

            samples.append(
                {
                    "target_latent": d["target_latent"].to(torch.bfloat16),
                    "ref_states": ref_states,
                    "ref_shape": (Hr, Wr),
                    "edge_weight": d["edge_weight"].to(torch.float32),
                    "txt_states": txt_states,
                }
            )
    torch.cuda.empty_cache()
    print(f"[OK] Prepared {len(samples)} extraction training samples in memory.")

    # 5. Inject LoRA (warm-started from flux3_klein_transparent_lora.safetensors)
    lora_modules = inject_extract_lora(
        model, rank=rank, alpha=alpha, device=device, init_lora_path=init_lora_path
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[OK] Injected Extraction LoRA (rank={rank}, alpha={alpha}) into {len(lora_modules)} layers ({num_trainable:,} params)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    warmup_steps = min(40, max(10, steps // 20))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, steps - warmup_steps))
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\n=================================================================")
    print(f"Starting FLUX 3 Image Klein Foreground-Extraction LoRA Training ({steps} steps)")
    print(f"LR: {lr}, Rank: {rank}, Grad Accum: {grad_accum}, Modules: {len(lora_modules)}")
    print("=================================================================")

    rng = random.Random(42)
    t_start = time.time()
    running_loss = 0.0
    running_mae = 0.0
    running_sharp = 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss_val = 0.0
        step_mae_val = 0.0
        step_sharp_val = 0.0

        for _ in range(grad_accum):
            item = samples[rng.randrange(len(samples))]
            x0 = item["target_latent"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            ref_states = item["ref_states"].to(device=device, dtype=torch.bfloat16)
            ref_shape = item["ref_shape"]
            ew = item["edge_weight"].unsqueeze(0).to(device=device, dtype=torch.float32)

            # 45% specific caption, 45% generic extraction prompt, 10% unconditional
            r_val = rng.random()
            if r_val < 0.10:
                txt_states = uncond_txt_states.to(device=device, dtype=torch.bfloat16)
            elif r_val < 0.55:
                txt_states = generic_txt_states.to(device=device, dtype=torch.bfloat16)
            else:
                txt_states = item["txt_states"].to(device=device, dtype=torch.bfloat16)

            t = sample_flux3_timestep(1, device=device, shift=FLUX3_SHIFT)
            noise = torch.randn_like(x0)
            tb = t.to(dtype=torch.bfloat16).view(-1, 1, 1, 1)
            noisy_latents = (1.0 - tb) * x0 + tb * noise
            target_v = (noise - x0).float()

            v_pred = forward_flux3_extract_trainable(model, noisy_latents, t, ref_states, ref_shape, txt_states)

            spatial_weight = 1.0 + (edge_multiplier - 1.0) * ew
            total_weight = spatial_weight * alpha_ch_weight
            mae_loss = ((v_pred.float() - target_v).abs() * total_weight).mean()
            sharp_loss = compute_sharpness_loss(
                v_pred, noisy_latents, x0, t, alpha_ch_weight, sharpness_multiplier=sharpness_multiplier
            )
            loss = (mae_loss + sharp_loss) / float(grad_accum)
            loss.backward()

            step_loss_val += loss.item()
            step_mae_val += mae_loss.item() / float(grad_accum)
            step_sharp_val += sharp_loss.item() / float(grad_accum)

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss = 0.9 * running_loss + 0.1 * step_loss_val if step > 1 else step_loss_val
        running_mae = 0.9 * running_mae + 0.1 * step_mae_val if step > 1 else step_mae_val
        running_sharp = 0.9 * running_sharp + 0.1 * step_sharp_val if step > 1 else step_sharp_val

        if step == 1 or step % 10 == 0 or step == steps:
            elapsed = time.time() - t_start
            it_per_s = step / max(elapsed, 1e-3)
            eta_min = (steps - step) / max(it_per_s, 1e-5) / 60.0
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"Step [{step:4d}/{steps}] | Loss: {running_loss:.4f} "
                f"(MAE: {running_mae:.4f}, Sharp: {running_sharp:.4f}) | "
                f"LR: {cur_lr:.2e} | {it_per_s:.2f} step/s | ETA: {eta_min:.1f}m",
                flush=True,
            )

        if save_every > 0 and step % save_every == 0 and step < steps:
            ckpt_name = output_name.replace(".safetensors", f"_step{step:04d}.safetensors")
            save_comfy_lora(lora_modules, os.path.join(loras_dir, ckpt_name))

    final_path = os.path.join(loras_dir, output_name)
    save_comfy_lora(lora_modules, final_path)
    print(f"\n[OK] Extraction LoRA Training finished! Saved to:\n   {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--output_name", type=str, default="flux3_klein_extract_lora.safetensors")
    args = parser.parse_args()

    train_extract(
        steps=args.steps,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        grad_accum=args.grad_accum,
        save_every=args.save_every,
        output_name=args.output_name,
    )
