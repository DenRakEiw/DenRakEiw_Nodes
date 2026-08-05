"""
TransparentVAE module for Flux LayerDiffuse integration in ComfyUI
Based on: https://github.com/RedAIGC/Flux-version-LayerDiffuse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
try:
    import safetensors.torch as sf
except ImportError:
    sf = None
from typing import Optional, Tuple
try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    # Create dummy classes for compatibility
    class ConfigMixin:
        pass
    class ModelMixin:
        pass
    def register_to_config(func):
        return func

from tqdm import tqdm


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 4 (RGBA)
class UNet1024(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,  # RGBA output
        down_block_types: Tuple[str] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (32, 32, 64, 128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 4,
        norm_eps: float = 1e-5,
        latent_c: int = 16,  # Flux latent channels
    ):
        super().__init__()

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.latent_conv_in = zero_module(nn.Conv2d(latent_c, block_out_channels[2], kernel_size=1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        if not DIFFUSERS_AVAILABLE:
            # Fallback to simple architecture if diffusers not available
            print("Warning: diffusers not available, using simplified decoder")
            return

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift="default",
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=None,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift="default",
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, latent):
        if not DIFFUSERS_AVAILABLE:
            # Simple fallback
            return torch.cat([x, torch.ones_like(x[:, :1])], dim=1)  # RGB + Alpha=1

        sample_latent = self.latent_conv_in(latent)
        sample = self.conv_in(x)
        emb = None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if i == 3:
                sample = sample + sample_latent
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        sample = self.mid_block(sample, emb)

        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples, emb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class LatentTransparencyOffsetEncoder(torch.nn.Module):
    def __init__(self, latent_c=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            zero_module(torch.nn.Conv2d(256, latent_c, kernel_size=3, padding=1, stride=1)),
        )

    def __call__(self, x):
        return self.blocks(x)


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2


def build_alpha_pyramid(color, alpha, dk=1.2):
    """Written by lvmin at Stanford"""
    pyramid = []
    current_premultiplied_color = color * alpha
    current_alpha = alpha
    while True:
        pyramid.append((current_premultiplied_color, current_alpha))
        H, W, C = current_alpha.shape
        if min(H, W) == 1:
            break
        current_premultiplied_color = cv2.resize(current_premultiplied_color, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)
        current_alpha = cv2.resize(current_alpha, (int(W / dk), int(H / dk)), interpolation=cv2.INTER_AREA)[:, :, None]
    return pyramid[::-1]


def pad_rgb(np_rgba_hwc_uint8):
    """Written by lvmin at Stanford"""
    np_rgba_hwc = np_rgba_hwc_uint8.astype(np.float32)
    pyramid = build_alpha_pyramid(color=np_rgba_hwc[..., :3], alpha=np_rgba_hwc[..., 3:])
    top_c, top_a = pyramid[0]
    fg = np.sum(top_c, axis=(0, 1), keepdims=True) / np.sum(top_a, axis=(0, 1), keepdims=True).clip(1e-8, 1e32)
    for layer_c, layer_a in pyramid:
        layer_h, layer_w, _ = layer_c.shape
        fg = cv2.resize(fg, (layer_w, layer_h), interpolation=cv2.INTER_LINEAR)
        fg = layer_c + fg * (1.0 - layer_a)
    return fg


def dist_sample_deterministic(dist, perturbation):
    """Modified from diffusers.models.autoencoders.vae.DiagonalGaussianDistribution.sample()"""
    if hasattr(dist, 'mean') and hasattr(dist, 'std'):
        x = dist.mean + dist.std * perturbation.to(dist.std)
    else:
        # Fallback for simple tensors
        x = dist + perturbation.to(dist)
    return x


class TransparentVAE(torch.nn.Module):
    def __init__(self, sd_vae, dtype=torch.float16, encoder_file=None, decoder_file=None, alpha=300.0, latent_c=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype
        self.sd_vae = sd_vae
        if hasattr(self.sd_vae, 'to'):
            self.sd_vae.to(dtype=self.dtype)
        if hasattr(self.sd_vae, 'requires_grad_'):
            self.sd_vae.requires_grad_(False)

        self.encoder = LatentTransparencyOffsetEncoder(latent_c=latent_c)
        if encoder_file is not None and sf is not None:
            try:
                temp = sf.load_file(encoder_file)
                self.encoder.load_state_dict(temp, strict=True)
                del temp
            except Exception as e:
                print(f"Warning: Could not load encoder file {encoder_file}: {e}")
        self.encoder.to(dtype=self.dtype)

        self.alpha = alpha

        # Use UNet1024 decoder like in the original implementation
        self.decoder = UNet1024(in_channels=3, out_channels=4, latent_c=latent_c)
        if decoder_file is not None:
            try:
                temp = sf.load_file(decoder_file)
                self.decoder.load_state_dict(temp, strict=True)
                del temp
                print(f"✓ Loaded decoder weights from {decoder_file}")
            except Exception as e:
                print(f"Warning: Could not load decoder file {decoder_file}: {e}")
        self.decoder.to(dtype=self.dtype)
        self.latent_c = latent_c

    def sd_decode(self, latent):
        decoded = self.sd_vae.decode(latent)
        # Handle both ComfyUI VAE (direct tensor) and diffusers VAE (.sample attribute)
        if hasattr(decoded, 'sample'):
            return decoded.sample
        else:
            return decoded

    def decode(self, latent, aug=True):
        decoded = self.sd_vae.decode(latent)
        # Handle both ComfyUI VAE (direct tensor) and diffusers VAE (.sample attribute)
        if hasattr(decoded, 'sample'):
            origin_pixel = decoded.sample
        else:
            origin_pixel = decoded
        origin_pixel = (origin_pixel * 0.5 + 0.5)

        if not aug:
            y = self.decoder(origin_pixel.to(self.dtype), latent.to(self.dtype))
            return origin_pixel, y

        list_y = []
        for i in range(int(latent.shape[0])):
            y = self.estimate_augmented(origin_pixel[i:i + 1].to(self.dtype), latent[i:i + 1].to(self.dtype))
            list_y.append(y)
        y = torch.concat(list_y, dim=0)
        return origin_pixel, y

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def estimate_augmented(self, pixel, latent):
        args = [
            [False, 0], [False, 1], [False, 2], [False, 3],
            [True, 0], [True, 1], [True, 2], [True, 3],
        ]
        result = []
        for flip, rok in args:  # Removed tqdm for now to avoid import issues
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()
            if flip:
                feed_pixel = torch.flip(feed_pixel, dims=(3,))
                feed_latent = torch.flip(feed_latent, dims=(3,))
            feed_pixel = torch.rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = torch.rot90(feed_latent, k=rok, dims=(2, 3))
            eps = self.decoder(feed_pixel, feed_latent).clip(0, 1)
            eps = torch.rot90(eps, k=-rok, dims=(2, 3))
            if flip:
                eps = torch.flip(eps, dims=(3,))
            result += [eps]
        result = torch.stack(result, dim=0)
        median = torch.median(result, dim=0).values
        return median

    def to(self, device):
        super().to(device)
        if hasattr(self.sd_vae, 'to'):
            self.sd_vae.to(device)
        return self

    def parameters(self):
        # Return both our parameters and VAE parameters
        params = list(super().parameters())
        if hasattr(self.sd_vae, 'parameters'):
            params.extend(list(self.sd_vae.parameters()))
        return iter(params)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict with proper handling"""
        try:
            # Filter state dict for our model
            our_state_dict = {}
            
            for key, value in state_dict.items():
                if any(prefix in key for prefix in ['encoder', 'decoder']):
                    our_state_dict[key] = value
            
            # Load our layers
            if our_state_dict:
                missing_keys, unexpected_keys = super().load_state_dict(our_state_dict, strict=False)
            else:
                missing_keys, unexpected_keys = [], []
            
            return missing_keys, unexpected_keys
            
        except Exception as e:
            print(f"Warning: Could not load some state dict keys: {e}")
            return [], []
