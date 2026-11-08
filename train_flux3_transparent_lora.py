#!/usr/bin/env python3
"""
Standalone LoRA Trainer for FLUX 3 Image Klein (5.6B) — Text-to-Transparent-Image.
Trains a ComfyUI-compatible LoRA on top of `flux3-image-klein-base-bf16.safetensors`
and the 4-channel RGBA VAE (`flux2_3_rgba_vae.safetensors` / trmz/flux2-klein-alpha).

Features:
- Premultiplied RGB ([0, 0, 0, 0] transparent background) matching trmz/flux2-klein-alpha
- Alpha-Channel Latent Sensitivity Weighting (boosts the exact latent channels encoding Alpha)
- Edge-Weighted MAE Loss (alpha_edge_multiplier = 3.0)
- Spatial Sharpness Gradient Loss (sharpness_multiplier = 0.25)
- 15% Unconditional ("") Caption Dropout so CFG=1.0..4.0 in KSampler stays 100% transparent
- Adapts emb_in.image, final_layer.image.linear, modulations, content_mode_blocks.image & single_blocks
"""

import argparse
import gc
import glob
import math
import os
import random
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import Image
import safetensors.torch as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import comfy.model_management
import comfy.ops
import comfy.sd
import comfy.utils
from comfy.ldm.flux3.model import (
    Flux3,
    modulated_norm,
    AttentionTensorContainer,
    optimized_attention,
    timestep_embedding,
)
from flux_rgba_vae_nodes import load_flux23_rgba_vae, DEFAULT_RGBA_VAE_FILENAME

FLUX3_SHIFT = 1.93586  # mu = ln(6.93) from comfy/supported_models.py


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for ComfyUI Linear layers."""

    def __init__(self, base_layer: nn.Module, rank: int = 32, alpha: float = 32.0, device=None):
        super().__init__()
        self.base = base_layer
        self.weight = base_layer.weight
        self.bias = getattr(base_layer, "bias", None)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.comfy_force_forward = True
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / float(rank)

        self.lora_down = nn.Linear(self.in_features, rank, bias=False, device=device, dtype=torch.float32)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(x.float())) * self.scale
        return base_out + lora_out.to(base_out.dtype)


def trainable_block_forward(block, x, spans, pe, transformer_options={}):
    """Out-of-place differentiable forward for FLUX 3 Block (supports gradient checkpointing)."""
    B, N, _ = x.shape
    if len(spans) == 1:
        mod = spans[0][2]
        x_mod = modulated_norm(x, block.pre_norm, mod.scale, mod.shift)
    else:
        chunks = [modulated_norm(x[:, s:e], block.pre_norm, m.scale, m.shift) for s, e, m in spans]
        x_mod = torch.cat(chunks, dim=1)

    if block.fused_qkv:
        q, k, v = block.qkv(x_mod).chunk(3, dim=-1)
    else:
        q, k, v = block.q_proj(x_mod), block.k_proj(x_mod), block.v_proj(x_mod)
    mlp = block.mlp_in(x_mod)
    q, k = block.norm(q.view(B, N, block.num_heads, -1), k.view(B, N, block.num_heads, -1), pe)
    q, k, v = (
        AttentionTensorContainer(q.flatten(2)),
        AttentionTensorContainer(k.flatten(2)),
        AttentionTensorContainer(v),
    )
    out = block.attn_out(
        optimized_attention(
            q,
            k,
            v,
            block.num_heads,
            transformer_options=transformer_options,
            preferred_attention=block.comfy_attention,
        )
    )
    out = comfy.ops.linear_input_act(
        block.mlp_out, mlp, "swiglu", residual=out, residual_scale=out.new_ones(out.shape[-1])
    )

    if len(spans) == 1:
        return x + out * spans[0][2].gate
    return torch.cat([x[:, s:e] + out[:, s:e] * m.gate for s, e, m in spans], dim=1)


def trainable_flux3_forward(model: Flux3, x: torch.Tensor, timestep: torch.Tensor, txt_states: torch.Tensor) -> torch.Tensor:
    """Out-of-place differentiable forward for FLUX 3 using precomputed t=0 txt_states."""
    B, _, H, W = x.shape
    dtype, device = x.dtype, x.device

    vec = model.time_in(timestep_embedding(torch.cat([timestep, timestep.new_zeros(1)]).to(dtype), 256))
    vec_img, vec_zero = vec[:-1], vec[-1:]
    early = model.early_stream_modulations["image"](vec_img)[0]

    img = model.emb_in["image"](x.flatten(2).transpose(1, 2))
    parts = [
        (img, "image", model.image_ids(H, W, 0, device)),
        (txt_states, "txt", model.text_ids(txt_states.shape[1], device)),
    ]
    pe = model.rope(torch.cat([ids for _, _, ids in parts]))
    spans, start = [], 0
    for tokens, mode, _ in parts:
        spans.append(
            (
                start,
                start + tokens.shape[1],
                model.single_stream_modulations[mode](vec_img if mode == "image" else vec_zero)[0],
            )
        )
        start += tokens.shape[1]

    pe_img = pe[:, : H * W]
    for block in model.content_mode_blocks["image"]:
        img = checkpoint(trainable_block_forward, block, img, [(0, H * W, early)], pe_img, use_reentrant=False)

    joint = torch.cat([img, txt_states], dim=1)
    for block in model.single_blocks:
        joint = checkpoint(trainable_block_forward, block, joint, spans, pe, use_reentrant=False)

    out = model.final_layer["image"](joint[:, : H * W], vec_img)
    return out.transpose(1, 2).reshape(B, model.in_channels, H, W)


def ensure_premul_latents_and_alpha_weights(
    dataset_dir: str,
    cache_files: list[str],
    vae_path: str,
    alpha_channel_boost: float = 3.0,
) -> torch.Tensor:
    """Ensure all cached latents are encoded with Premultiplied RGB ([0,0,0,0] background)
    and return a [1, 128, 1, 1] per-channel loss weight tensor boosting the Alpha-sensitive channels."""
    images_dir = os.path.join(dataset_dir, "images")
    needs_reencode = []
    for pf in cache_files:
        d = torch.load(pf, map_location="cpu")
        if not d.get("premul_rgba", False):
            needs_reencode.append(pf)

    print(f"Loading RGBA VAE from: {vae_path}")
    rgba_vae = load_flux23_rgba_vae(vae_path, dtype_str="bfloat16")

    # Compute 128-channel Alpha sensitivity profile from RGBA VAE
    with torch.no_grad():
        img_trans = torch.zeros(1, 256, 256, 4)
        img_opaque = torch.zeros(1, 256, 256, 4)
        img_opaque[..., 3] = 1.0
        z_t = rgba_vae.encode(img_trans).float()
        z_o = rgba_vae.encode(img_opaque).float()
        ch_diff = (z_o - z_t).abs().mean(dim=(0, 2, 3))  # [128]
        ch_norm = ch_diff / ch_diff.max().clamp(min=1e-5)
        alpha_ch_weight = 1.0 + alpha_channel_boost * ch_norm
        alpha_ch_weight = (alpha_ch_weight / alpha_ch_weight.mean()).view(1, 128, 1, 1)
        print(
            f"✓ Computed 128-ch Alpha sensitivity weights "
            f"(min={alpha_ch_weight.min().item():.2f}, max={alpha_ch_weight.max().item():.2f})"
        )

    if needs_reencode:
        print(f"Re-encoding {len(needs_reencode)} samples with Premultiplied RGB ([0,0,0,0] background)...")
        with torch.no_grad():
            for i, pf in enumerate(needs_reencode):
                d = torch.load(pf, map_location="cpu")
                base_name = os.path.splitext(os.path.basename(pf))[0]
                png_path = os.path.join(images_dir, f"{base_name}.png")
                rgba_u8 = np.array(Image.open(png_path).convert("RGBA"), dtype=np.float32) / 255.0
                alpha_ch = rgba_u8[..., 3:4]
                premul_rgb = np.clip(rgba_u8[..., :3] * alpha_ch, 0.0, 1.0)
                premul_rgba = np.concatenate([premul_rgb, alpha_ch], axis=-1)
                rgba_tensor = torch.from_numpy(premul_rgba).unsqueeze(0)
                latent = rgba_vae.encode(rgba_tensor)[0].detach().to(torch.bfloat16).cpu()
                d["latent"] = latent
                d["premul_rgba"] = True
                torch.save(d, pf)
                if (i + 1) % 100 == 0 or (i + 1) == len(needs_reencode):
                    print(f"  [{i + 1}/{len(needs_reencode)}] Re-encoded Premultiplied RGBA latent")
    else:
        print(f"✓ All {len(cache_files)} cache files already use Premultiplied RGBA latents.")

    del rgba_vae
    gc.collect()
    torch.cuda.empty_cache()
    return alpha_ch_weight


def cache_text_embeddings(dataset_dir: str, cache_files: list[str], clip_path: str) -> dict:
    """Encode captions (and unconditional '' prompt) with f3_qwen3vl_4b.safetensors (CLIPType.FLUX3)."""
    uncond_cache_path = os.path.join(dataset_dir, "uncond_context.pt")
    missing = []
    for pf in cache_files:
        data = torch.load(pf, map_location="cpu")
        if not data.get("norm_scale_fixed", False) or ("context" not in data and "txt_states" not in data):
            missing.append(pf)

    need_uncond = True
    if os.path.exists(uncond_cache_path):
        ud = torch.load(uncond_cache_path, map_location="cpu")
        if ud.get("norm_scale_fixed", False) and "txt_states" in ud and not torch.isnan(ud["txt_states"]).any():
            need_uncond = False

    if not missing and not need_uncond:
        print(f"✓ All {len(cache_files)} cache files and uncond prompt already have valid text embeddings.")
        return torch.load(uncond_cache_path, map_location="cpu")

    print(f"Loading Qwen3-VL-4B (CLIPType.FLUX3) for text encoding (missing={len(missing)}, uncond={need_uncond})...")
    clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.FLUX3)

    if need_uncond:
        with torch.no_grad():
            tokens = clip.tokenize("")
            cond = clip.encode_from_tokens_scheduled(tokens)
            uncond_ctx = cond[0][0].to(torch.bfloat16).cpu()
            torch.save({"context": uncond_ctx}, uncond_cache_path)
            print(f"✓ Encoded unconditional '' prompt -> shape {tuple(uncond_ctx.shape)}")

    for i, pf in enumerate(missing):
        data = torch.load(pf, map_location="cpu")
        caption = data["caption"]
        tokens = clip.tokenize(caption)
        with torch.no_grad():
            cond = clip.encode_from_tokens_scheduled(tokens)
            context = cond[0][0].to(torch.bfloat16).cpu()
        data["context"] = context
        if "txt_states" in data:
            del data["txt_states"]
        torch.save(data, pf)
        if (i + 1) % 100 == 0 or (i + 1) == len(missing):
            print(f"  [{i + 1}/{len(missing)}] Encoded caption -> shape {tuple(context.shape)}")

    del clip
    gc.collect()
    torch.cuda.empty_cache()
    return torch.load(uncond_cache_path, map_location="cpu")


def inject_lora(model: Flux3, rank: int, alpha: float, device: torch.device) -> dict[str, LoRALinear]:
    """Inject LoRA into emb_in['image'], final_layer['image'].linear, modulations,
    and all Linear layers of content_mode_blocks['image'] and single_blocks."""
    lora_modules: dict[str, LoRALinear] = {}
    target_linear_names = ["q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out"]

    # 1. Adapt 128-channel RGBA latent input/output projections
    if "image" in model.emb_in:
        base_in = model.emb_in["image"]
        lora_in = LoRALinear(base_in, rank=rank, alpha=alpha, device=device)
        model.emb_in["image"] = lora_in
        lora_modules["emb_in.image"] = lora_in

    if "image" in model.final_layer and hasattr(model.final_layer["image"], "linear"):
        base_out = model.final_layer["image"].linear
        lora_out = LoRALinear(base_out, rank=rank, alpha=alpha, device=device)
        model.final_layer["image"].linear = lora_out
        lora_modules["final_layer.image.linear"] = lora_out

    if "image" in model.final_layer and hasattr(model.final_layer["image"], "adaLN_modulation"):
        ada_lin = model.final_layer["image"].adaLN_modulation[1]
        lora_ada = LoRALinear(ada_lin, rank=rank, alpha=alpha, device=device)
        model.final_layer["image"].adaLN_modulation[1] = lora_ada
        lora_modules["final_layer.image.adaLN_modulation.1"] = lora_ada

    if "image" in model.early_stream_modulations and hasattr(model.early_stream_modulations["image"], "lin"):
        emod = model.early_stream_modulations["image"].lin
        lora_emod = LoRALinear(emod, rank=rank, alpha=alpha, device=device)
        model.early_stream_modulations["image"].lin = lora_emod
        lora_modules["early_stream_modulations.image.lin"] = lora_emod

    if "image" in model.single_stream_modulations and hasattr(model.single_stream_modulations["image"], "lin"):
        smod = model.single_stream_modulations["image"].lin
        lora_smod = LoRALinear(smod, rank=rank, alpha=alpha, device=device)
        model.single_stream_modulations["image"].lin = lora_smod
        lora_modules["single_stream_modulations.image.lin"] = lora_smod

    # 2. Adapt all image blocks and joint single_blocks
    for idx, block in enumerate(model.content_mode_blocks["image"]):
        for lname in target_linear_names:
            if hasattr(block, lname):
                base_layer = getattr(block, lname)
                lora_layer = LoRALinear(base_layer, rank=rank, alpha=alpha, device=device)
                setattr(block, lname, lora_layer)
                key = f"content_mode_blocks.image.{idx}.{lname}"
                lora_modules[key] = lora_layer

    for idx, block in enumerate(model.single_blocks):
        for lname in target_linear_names:
            if hasattr(block, lname):
                base_layer = getattr(block, lname)
                lora_layer = LoRALinear(base_layer, rank=rank, alpha=alpha, device=device)
                setattr(block, lname, lora_layer)
                key = f"single_blocks.{idx}.{lname}"
                lora_modules[key] = lora_layer

    return lora_modules


def save_comfy_lora(lora_modules: dict[str, LoRALinear], save_path: str, metadata: dict = None):
    """Save LoRA weights in native ComfyUI format (`diffusion_model.<path>.lora_down.weight`)."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    state_dict = {}
    for key, mod in lora_modules.items():
        prefix = f"diffusion_model.{key}"
        state_dict[f"{prefix}.lora_down.weight"] = mod.lora_down.weight.detach().to(torch.bfloat16).cpu().contiguous()
        state_dict[f"{prefix}.lora_up.weight"] = mod.lora_up.weight.detach().to(torch.bfloat16).cpu().contiguous()
        state_dict[f"{prefix}.alpha"] = torch.tensor(mod.alpha, dtype=torch.float32)

    str_meta = {k: str(v) for k, v in (metadata or {}).items()}
    sf.save_file(state_dict, save_path, metadata=str_meta)
    print(f"💾 Saved ComfyUI LoRA ({len(lora_modules)} modules) -> {save_path}")


def sample_flux3_timestep(batch_size: int, device: torch.device, shift: float = FLUX3_SHIFT) -> torch.Tensor:
    """Sample logit-normal timestep in (0, 1) and apply FLUX 3 shift (mu = 1.93586)."""
    u = torch.randn(batch_size, device=device, dtype=torch.float32)
    u = torch.sigmoid(u).clamp(1e-4, 1.0 - 1e-4)
    exp_mu = math.exp(shift)
    t = exp_mu / (exp_mu + (1.0 / u - 1.0))
    return t.clamp(1e-4, 1.0 - 1e-4)


def compute_sharpness_loss(
    v_pred: torch.Tensor,
    noisy_latents: torch.Tensor,
    target_latents: torch.Tensor,
    t: torch.Tensor,
    alpha_ch_weight: torch.Tensor,
    sharpness_multiplier: float = 0.25,
) -> torch.Tensor:
    """Spatial gradient sharpness loss from trmz/flux2-klein-alpha §7, weighted by alpha sensitivity."""
    if sharpness_multiplier <= 0.0:
        return torch.tensor(0.0, device=v_pred.device)

    tv = t.view(-1, 1, 1, 1).clamp(min=1e-3)
    x0_pred = noisy_latents.float() - tv * v_pred.float()
    x0_true = target_latents.float()

    pred_gx = x0_pred[:, :, :, 1:] - x0_pred[:, :, :, :-1]
    pred_gy = x0_pred[:, :, 1:, :] - x0_pred[:, :, :-1, :]
    true_gx = x0_true[:, :, :, 1:] - x0_true[:, :, :, :-1]
    true_gy = x0_true[:, :, 1:, :] - x0_true[:, :, :-1, :]

    gx_err = ((pred_gx - true_gx).abs() * alpha_ch_weight).mean(dim=(1, 2, 3))
    gy_err = ((pred_gy - true_gy).abs() * alpha_ch_weight).mean(dim=(1, 2, 3))
    per_sample = (gx_err + gy_err) * (1.0 - t)
    return per_sample.mean() * sharpness_multiplier


def train(
    dataset_dir: str = "datasets/flux3_transparent",
    steps: int = 1000,
    lr: float = 1e-4,
    rank: int = 32,
    alpha: float = 32.0,
    grad_accum: int = 2,
    edge_multiplier: float = 3.0,
    sharpness_multiplier: float = 0.25,
    alpha_channel_boost: float = 3.0,
    caption_dropout: float = 0.15,
    save_every: int = 250,
    output_name: str = "flux3_klein_transparent_lora.safetensors",
):
    comfy.model_management.in_training = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = os.path.join(dataset_dir, "cache")
    cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    if not cache_files:
        raise FileNotFoundError(
            f"No cached samples found in {cache_dir}. Run `python create_flux3_transparent_dataset.py` first!"
        )

    vae_path = os.path.join(COMFY_ROOT, "models", "vae", DEFAULT_RGBA_VAE_FILENAME)
    clip_path = os.path.join(COMFY_ROOT, "models", "text_encoders", "f3_qwen3vl_4b.safetensors")
    model_path = os.path.join(COMFY_ROOT, "models", "diffusion_models", "flux3-image-klein-base-bf16.safetensors")
    loras_dir = os.path.join(COMFY_ROOT, "models", "loras")
    os.makedirs(loras_dir, exist_ok=True)

    # 1. Ensure Premultiplied RGBA latents and compute 128-ch Alpha sensitivity weights
    alpha_ch_weight = ensure_premul_latents_and_alpha_weights(
        dataset_dir, cache_files, vae_path, alpha_channel_boost=alpha_channel_boost
    ).to(device=device, dtype=torch.float32)

    # 2. Ensure all captions + unconditional prompt are encoded
    uncond_data = cache_text_embeddings(dataset_dir, cache_files, clip_path)

    # 3. Load FLUX 3 Image Klein Base (5.6B)
    print(f"Loading FLUX 3 Image Klein Base from: {model_path}")
    raw_sd = comfy.utils.load_torch_file(model_path)
    # ComfyUI Flux.process_unet_state_dict maps '*.query_norm.scale' -> '*.query_norm.weight'
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
    print(f"✓ Loaded FLUX 3 Base on {device} (missing={len(missing)}, unexpected={len(unexpected)})")

    # 4. Precompute t=0 txt_states (including unconditional '' txt_states)
    print(f"Preparing {len(cache_files)} samples (computing t=0 txt_states)...")
    with torch.no_grad():
        if "txt_states" in uncond_data and uncond_data.get("norm_scale_fixed", False):
            uncond_txt_states = uncond_data["txt_states"].to(torch.bfloat16)
        else:
            uctx = uncond_data["context"].to(device=device, dtype=torch.bfloat16)
            uncond_txt_states, _ = model.preprocess_conds(uctx)
            uncond_txt_states = uncond_txt_states.detach().to(torch.bfloat16).cpu()
            uncond_data["txt_states"] = uncond_txt_states
            uncond_data["norm_scale_fixed"] = True
            torch.save(uncond_data, os.path.join(dataset_dir, "uncond_context.pt"))

        samples = []
        for i, pf in enumerate(cache_files):
            d = torch.load(pf, map_location="cpu")
            if "txt_states" in d and d.get("norm_scale_fixed", False):
                txt_states = d["txt_states"].to(torch.bfloat16)
            else:
                ctx = d["context"].to(device=device, dtype=torch.bfloat16)
                txt_states, _ = model.preprocess_conds(ctx)
                txt_states = txt_states.detach().to(torch.bfloat16).cpu()
                d["txt_states"] = txt_states
                d["norm_scale_fixed"] = True
                if "context" in d:
                    del d["context"]
                torch.save(d, pf)

            samples.append(
                {
                    "latent": d["latent"].to(torch.bfloat16),          # [128, H/16, W/16]
                    "edge_weight": d["edge_weight"].to(torch.float32), # [1, H/16, W/16]
                    "txt_states": txt_states,                          # [1, seq_len, 3072]
                    "caption": d.get("caption", ""),
                }
            )
    torch.cuda.empty_cache()
    print(f"✓ Prepared {len(samples)} training samples in memory.")

    # 5. Inject LoRA adapters
    lora_modules = inject_lora(model, rank=rank, alpha=alpha, device=device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"✓ Injected LoRA (rank={rank}, alpha={alpha}) into {len(lora_modules)} linear layers ({num_trainable:,} trainable params)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    warmup_steps = min(50, max(10, steps // 20))

    def lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        progress = float(step_idx - warmup_steps) / float(max(1, steps - warmup_steps))
        return 0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("\n" + "=" * 65)
    print(f"🚀 Starting FLUX 3 Image Klein Transparent LoRA Training ({steps} steps)")
    print(f"   LR: {lr}, Rank: {rank}, Grad Accum: {grad_accum}, Alpha Boost: {alpha_channel_boost}x, Dropout: {caption_dropout}")
    print("=" * 65)

    running_loss = 0.0
    running_mae = 0.0
    running_sharp = 0.0
    t_start = time.time()

    optimizer.zero_grad()

    for step in range(1, steps + 1):
        step_loss = 0.0
        step_mae = 0.0
        step_sharp = 0.0

        for _ in range(grad_accum):
            item = random.choice(samples)
            x0 = item["latent"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [1, 128, H/16, W/16]
            ew = item["edge_weight"].unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, 1, H/16, W/16]

            # 15% unconditional caption dropout so CFG (e.g. 3.5) preserves 0.0 background alpha
            if random.random() < caption_dropout:
                txt_states = uncond_txt_states.to(device=device, dtype=torch.bfloat16)
            else:
                txt_states = item["txt_states"].to(device=device, dtype=torch.bfloat16)

            # Optional horizontal flip augmentation (50% chance)
            if random.random() < 0.5:
                B, C, H, W = x0.shape
                x0_unpacked = x0.view(B, 32, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, 32, H * 2, W * 2)
                x0_unpacked = torch.flip(x0_unpacked, dims=(-1,))
                x0 = x0_unpacked.view(B, 32, H, 2, W, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, 128, H, W)
                ew = torch.flip(ew, dims=(-1,))

            t = sample_flux3_timestep(1, device=device, shift=FLUX3_SHIFT)
            noise = torch.randn_like(x0)
            t_view = t.view(-1, 1, 1, 1).to(torch.bfloat16)

            noisy_latents = (1.0 - t_view) * x0 + t_view * noise
            target_v = (noise - x0).float()

            pred_v = trainable_flux3_forward(model, noisy_latents, t.to(torch.bfloat16), txt_states)

            # Edge-weighted + Alpha-channel-weighted MAE loss
            mae_map = (pred_v.float() - target_v).abs()
            weighted_mae = (mae_map * ew * alpha_ch_weight).mean()

            # Spatial sharpness gradient loss
            sharp_loss = compute_sharpness_loss(
                pred_v, noisy_latents, x0, t, alpha_ch_weight, sharpness_multiplier=sharpness_multiplier
            )

            loss = (weighted_mae + sharp_loss) / float(grad_accum)
            loss.backward()

            step_loss += loss.item()
            step_mae += weighted_mae.item() / float(grad_accum)
            step_sharp += sharp_loss.item() / float(grad_accum)

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += step_loss
        running_mae += step_mae
        running_sharp += step_sharp

        if step % 10 == 0 or step == 1 or step == steps:
            denom = 10 if step > 1 and step % 10 == 0 else 1
            avg_loss = running_loss / denom if step > 1 else step_loss
            avg_mae = running_mae / denom if step > 1 else step_mae
            avg_sharp = running_sharp / denom if step > 1 else step_sharp
            if step % 10 == 0:
                running_loss = 0.0
                running_mae = 0.0
                running_sharp = 0.0
            elapsed = time.time() - t_start
            it_s = step / max(elapsed, 1e-3)
            eta_min = (steps - step) / max(it_s, 1e-5) / 60.0
            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"Step [{step:4d}/{steps}] | Loss: {avg_loss:.4f} (MAE: {avg_mae:.4f}, Sharp: {avg_sharp:.4f}) "
                f"| LR: {cur_lr:.2e} | {it_s:.2f} step/s | ETA: {eta_min:.1f}m"
            )

        if save_every > 0 and step % save_every == 0 and step < steps:
            ckpt_name = output_name.replace(".safetensors", f"_step{step:04d}.safetensors")
            ckpt_path = os.path.join(loras_dir, ckpt_name)
            save_comfy_lora(
                lora_modules,
                ckpt_path,
                metadata={
                    "model": "FLUX.3-Image-Klein-5.6B",
                    "task": "text-to-transparent-image",
                    "vae": "trmz/flux2-klein-alpha (flux2_3_rgba_vae.safetensors)",
                    "step": step,
                    "lora_rank": rank,
                    "lora_alpha": alpha,
                },
            )

    final_path = os.path.join(loras_dir, output_name)
    save_comfy_lora(
        lora_modules,
        final_path,
        metadata={
            "model": "FLUX.3-Image-Klein-5.6B",
            "task": "text-to-transparent-image",
            "vae": "trmz/flux2-klein-alpha (flux2_3_rgba_vae.safetensors)",
            "step": steps,
            "lora_rank": rank,
            "lora_alpha": alpha,
            "trigger_word": "transparent background",
        },
    )
    print(f"\n🎉 Training finished! Final LoRA saved to:\n   {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FLUX 3 Image Klein Transparent LoRA")
    parser.add_argument("--dataset_dir", type=str, default="datasets/flux3_transparent")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--edge_multiplier", type=float, default=3.0)
    parser.add_argument("--sharpness_multiplier", type=float, default=0.25)
    parser.add_argument("--alpha_channel_boost", type=float, default=3.0)
    parser.add_argument("--caption_dropout", type=float, default=0.15)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--output_name", type=str, default="flux3_klein_transparent_lora.safetensors")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset_dir,
        steps=args.steps,
        lr=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        grad_accum=args.grad_accum,
        edge_multiplier=args.edge_multiplier,
        sharpness_multiplier=args.sharpness_multiplier,
        alpha_channel_boost=args.alpha_channel_boost,
        caption_dropout=args.caption_dropout,
        save_every=args.save_every,
        output_name=args.output_name,
    )
