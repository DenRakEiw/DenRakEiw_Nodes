"""
Flux 2 / Flux 3 Klein 4-Channel RGBA VAE Nodes for ComfyUI
Based on AlphaVAE / trmz/flux2-klein-alpha (AutoencoderKLFlux2 widened to 4 RGBA channels).
Compatible with both FLUX.2 Klein (4B/9B) and FLUX 3 Image Klein (5.6B).
"""

import os
import torch
import numpy as np

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management as model_management

try:
    from .transparent_vae import pad_rgb
except ImportError:
    from transparent_vae import pad_rgb

HF_RGBA_VAE_URL = (
    "https://huggingface.co/trmz/flux2-klein-alpha/resolve/main/vae/diffusion_pytorch_model.safetensors"
)
ULTRA_RGBA_VAE_FILENAME = "flux3_ultra_rgba_vae.safetensors"
DEFAULT_RGBA_VAE_FILENAME = "flux2_3_rgba_vae.safetensors"


def load_flux23_rgba_vae(checkpoint_path: str, dtype_str: str = "auto") -> comfy.sd.VAE:
    """Load a 4-channel RGBA AutoencoderKLFlux2 (32 raw latent channels / 128 packed channels) into ComfyUI's VAE."""
    sd = comfy.utils.load_torch_file(checkpoint_path)

    # Detect input/output channels from state dict (handles converted or raw diffusers keys)
    conv_in_key = "encoder.conv_in.weight"
    conv_out_key = "decoder.conv_out.weight"
    in_channels = int(sd[conv_in_key].shape[1]) if conv_in_key in sd else 4
    out_ch = int(sd[conv_out_key].shape[0]) if conv_out_key in sd else 4
    z_channels = int(sd["decoder.conv_in.weight"].shape[1]) if "decoder.conv_in.weight" in sd else 32

    ddconfig = {
        "double_z": True,
        "z_channels": z_channels,
        "resolution": 256,
        "in_channels": in_channels,
        "out_ch": out_ch,
        "ch": 128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0,
        "batch_norm_latent": "bn.running_mean" in sd,
        "fused_norm": False,
    }

    dtype_map = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, None)

    config = {"params": {"ddconfig": ddconfig, "embed_dim": z_channels}}
    vae = comfy.sd.VAE(sd=sd, config=config, dtype=torch_dtype)

    if ddconfig["batch_norm_latent"]:
        vae.downscale_ratio = 16
        vae.upscale_ratio = 16
        vae.latent_channels = z_channels * 4  # 128 packed channels
    else:
        vae.downscale_ratio = 8
        vae.upscale_ratio = 8
        vae.latent_channels = z_channels

    vae.output_channels = out_ch
    vae.pad_channel_value = 1.0
    vae.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    return vae


# ----------------------------------------------------------------------------
# Hook vanilla ComfyUI `comfy.sd.VAE.__init__` so the built-in `VAELoader` node
# can also load 4-channel RGBA AutoencoderKLFlux2 VAEs without size mismatch!
# ----------------------------------------------------------------------------
if not getattr(comfy.sd.VAE, "_dre_rgba_vae_hooked", False):
    _orig_comfy_vae_init = comfy.sd.VAE.__init__

    def _dre_patched_vae_init(self, sd=None, device=None, config=None, dtype=None, metadata=None, no_init=False):
        is_4ch_flux_vae = False
        if not no_init and config is None and sd is not None:
            conv_in_w = sd.get("encoder.conv_in.weight", None)
            conv_out_w = sd.get("decoder.conv_out.weight", None)
            if (
                conv_in_w is not None
                and conv_out_w is not None
                and conv_in_w.ndim == 4
                and (conv_in_w.shape[1] == 4 or conv_out_w.shape[0] == 4)
                and "bn.running_mean" in sd
            ):
                is_4ch_flux_vae = True
                in_channels = int(conv_in_w.shape[1])
                out_ch = int(conv_out_w.shape[0])
                z_channels = int(sd["decoder.conv_in.weight"].shape[1]) if "decoder.conv_in.weight" in sd else 32
                ddconfig = {
                    "double_z": True,
                    "z_channels": z_channels,
                    "resolution": 256,
                    "in_channels": in_channels,
                    "out_ch": out_ch,
                    "ch": 128,
                    "ch_mult": [1, 2, 4, 4],
                    "num_res_blocks": 2,
                    "attn_resolutions": [],
                    "dropout": 0.0,
                    "batch_norm_latent": True,
                    "fused_norm": False,
                }
                config = {"params": {"ddconfig": ddconfig, "embed_dim": z_channels}}

        _orig_comfy_vae_init(self, sd=sd, device=device, config=config, dtype=dtype, metadata=metadata, no_init=no_init)

        if is_4ch_flux_vae:
            z_ch = config["params"]["embed_dim"]
            self.downscale_ratio = 16
            self.upscale_ratio = 16
            self.latent_channels = z_ch * 4
            self.output_channels = config["params"]["ddconfig"]["out_ch"]
            self.pad_channel_value = 1.0
            self.working_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    comfy.sd.VAE.__init__ = _dre_patched_vae_init
    comfy.sd.VAE._dre_rgba_vae_hooked = True


def apply_alpha_bleed_batch(rgba_tensor: torch.Tensor) -> torch.Tensor:
    """Fill RGB values under transparent pixels using pyramid alpha bleeding (pad_rgb)
    so zero/black transparent backgrounds do not cause dark edge halos during VAE encoding.

    rgba_tensor: [B, H, W, 4] float tensor in [0, 1].
    """
    out_list = []
    rgba_np = rgba_tensor.detach().cpu().numpy()
    for i in range(rgba_np.shape[0]):
        img = np.clip(rgba_np[i], 0.0, 1.0)
        alpha = img[..., 3:4]
        # Only run pyramid bleed if there are transparent pixels and some opaque pixels
        if alpha.min() < 0.999 and alpha.max() > 1e-4:
            bled_rgb = np.clip(pad_rgb(img), 0.0, 1.0)
            img = np.concatenate([bled_rgb, alpha], axis=-1)
        out_list.append(img)
    out_np = np.stack(out_list, axis=0).astype(np.float32)
    return torch.from_numpy(out_np).to(device=rgba_tensor.device, dtype=rgba_tensor.dtype)


class Flux23RGBAVAELoader:
    """Loads the 4-channel RGBA VAE (AutoencoderKLFlux2) for FLUX.2 Klein and FLUX 3 Image Klein."""

    @classmethod
    def INPUT_TYPES(cls):
        vae_files = list(folder_paths.get_filename_list("vae"))
        preferred = []
        for candidate in (ULTRA_RGBA_VAE_FILENAME, DEFAULT_RGBA_VAE_FILENAME):
            if candidate in vae_files:
                preferred.append(candidate)
        if not preferred:
            preferred.append(DEFAULT_RGBA_VAE_FILENAME)
        vae_files = preferred + [f for f in vae_files if f not in preferred]
        return {
            "required": {
                "vae_name": (vae_files,),
                "dtype": (["auto", "bfloat16", "float16", "float32"], {"default": "auto"}),
                "auto_download_if_missing": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("rgba_vae",)
    FUNCTION = "load_vae"
    CATEGORY = "denrakeiw/flux"

    def load_vae(self, vae_name: str, dtype: str = "auto", auto_download_if_missing: bool = True):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        if vae_path is None or not os.path.exists(vae_path):
            target_dir = os.path.join(folder_paths.models_dir, "vae")
            os.makedirs(target_dir, exist_ok=True)
            candidate = os.path.join(target_dir, vae_name)
            if os.path.exists(candidate):
                vae_path = candidate
            elif auto_download_if_missing:
                import requests
                print(f"Downloading Flux 2/3 RGBA VAE from {HF_RGBA_VAE_URL} -> {candidate} ...")
                with requests.get(HF_RGBA_VAE_URL, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(candidate, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                vae_path = candidate
                print(f"[Flux23RGBAVAE] Downloaded RGBA VAE: {vae_path}")
            else:
                raise FileNotFoundError(
                    f"Flux 2/3 RGBA VAE '{vae_name}' not found in models/vae/ and auto_download_if_missing=False."
                )

        vae = load_flux23_rgba_vae(vae_path, dtype_str=dtype)
        print(
            f"[Flux23RGBAVAE] Loaded Flux 2/3 RGBA VAE from {vae_path} "
            f"(latent_channels={vae.latent_channels}, output_channels={vae.output_channels})"
        )
        return (vae,)


class Flux23RGBAVAEDecode:
    """Decodes FLUX.2 / FLUX 3 Klein latents ([B, 128, H/16, W/16]) into 4-channel RGBA images,
    3-channel RGB images, and an alpha MASK."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgba_vae": ("VAE",),
                "samples": ("LATENT",),
            },
            "optional": {
                "tiled": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 256}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "rgb_image", "alpha_mask")
    FUNCTION = "decode"
    CATEGORY = "denrakeiw/flux"

    def decode(self, rgba_vae: comfy.sd.VAE, samples: dict, tiled: bool = False, tile_size: int = 1024):
        latent = samples["samples"]
        # Auto-enable 4-channel tiled decoding for high-res >= 1536px or if explicitly requested
        h_px = latent.shape[-2] * 16
        w_px = latent.shape[-1] * 16
        use_tiled = tiled or (max(h_px, w_px) >= 1536)
        with torch.no_grad():
            if use_tiled:
                c = rgba_vae.spacial_compression_decode()
                t_lat = max(32, tile_size // c)
                images = rgba_vae.decode_tiled(latent, tile_x=t_lat, tile_y=t_lat, overlap=max(8, t_lat // 4)).detach()
            else:
                images = rgba_vae.decode(latent).detach()
        if images.ndim == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        images = images.clamp(0.0, 1.0)

        if images.shape[-1] == 4:
            rgba_image = images
            rgb_image = images[..., :3]
            alpha_mask = images[..., 3]
        else:
            rgb_image = images[..., :3]
            alpha_mask = torch.ones(
                (images.shape[0], images.shape[1], images.shape[2]),
                dtype=images.dtype,
                device=images.device,
            )
            rgba_image = torch.cat([rgb_image, alpha_mask.unsqueeze(-1)], dim=-1)

        return (rgba_image, rgb_image, alpha_mask)


class Flux23RGBAVAEEncode:
    """Encodes RGBA (or RGB + optional MASK) images into FLUX.2 / FLUX 3 Klein 128-channel latents,
    with optional pyramid alpha-bleeding (pad_rgb) to prevent dark edge halos."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgba_vae": ("VAE",),
                "image": ("IMAGE",),
                "alpha_bleed": ("BOOLEAN", {"default": True}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "prepared_rgba")
    FUNCTION = "encode"
    CATEGORY = "denrakeiw/flux"

    def encode(
        self,
        rgba_vae: comfy.sd.VAE,
        image: torch.Tensor,
        alpha_bleed: bool = True,
        invert_mask: bool = False,
        mask: torch.Tensor = None,
    ):
        rgb = image[..., :3]
        if mask is not None:
            m = mask.to(device=image.device, dtype=image.dtype)
            if m.ndim == 2:
                m = m.unsqueeze(0)
            if m.shape[1:3] != image.shape[1:3]:
                m = torch.nn.functional.interpolate(
                    m.unsqueeze(1), size=(image.shape[1], image.shape[2]), mode="bilinear", align_corners=False
                ).squeeze(1)
            alpha = (1.0 - m) if invert_mask else m
            alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)
            if alpha.shape[0] != rgb.shape[0]:
                alpha = alpha.expand(rgb.shape[0], -1, -1, -1)
            rgba = torch.cat([rgb, alpha], dim=-1)
        elif image.shape[-1] == 4:
            rgba = image.clamp(0.0, 1.0)
        else:
            alpha = torch.ones(
                (rgb.shape[0], rgb.shape[1], rgb.shape[2], 1),
                dtype=rgb.dtype,
                device=rgb.device,
            )
            rgba = torch.cat([rgb, alpha], dim=-1)

        if alpha_bleed:
            rgba = apply_alpha_bleed_batch(rgba)

        with torch.no_grad():
            latent_tensor = rgba_vae.encode(rgba).detach()
        return ({"samples": latent_tensor}, rgba)


NODE_CLASS_MAPPINGS = {
    "Flux23RGBAVAELoader_DRE": Flux23RGBAVAELoader,
    "Flux23RGBAVAEDecode_DRE": Flux23RGBAVAEDecode,
    "Flux23RGBAVAEEncode_DRE": Flux23RGBAVAEEncode,
    "Flux23RGBAVAELoader": Flux23RGBAVAELoader,
    "Flux23RGBAVAEDecode": Flux23RGBAVAEDecode,
    "Flux23RGBAVAEEncode": Flux23RGBAVAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux23RGBAVAELoader_DRE": "🔧 Flux 2/3 RGBA VAE Loader *DRE",
    "Flux23RGBAVAEDecode_DRE": "🔍 Flux 2/3 RGBA VAE Decode *DRE",
    "Flux23RGBAVAEEncode_DRE": "📥 Flux 2/3 RGBA VAE Encode *DRE",
    "Flux23RGBAVAELoader": "🔧 Flux 2/3 RGBA VAE Loader *DRE",
    "Flux23RGBAVAEDecode": "🔍 Flux 2/3 RGBA VAE Decode *DRE",
    "Flux23RGBAVAEEncode": "📥 Flux 2/3 RGBA VAE Encode *DRE",
}
