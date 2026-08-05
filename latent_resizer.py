"""
Neural Network Latent Resizer Models

Contains the model architectures for latent upscaling.

Architecture based on Ttl's ComfyUi_NNLatentUpscale:
https://github.com/Ttl/ComfyUi_NNLatentUpscale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LatentResizer(nn.Module):
    """
    Universal Neural network for upscaling latents in diffusion models.
    Supports SD1.5, SDXL, Flux, and Wan2.2 architectures.
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        # Encoder layers - extract features from input latents
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling layers - reconstruct at higher resolution
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, out_channels, 3, padding=1),
        )
        
        # Skip connection for residual learning
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
        """
        Forward pass with configurable scale factor.

        Args:
            x: Input latent tensor [B, C, H, W]
            scale: Upscaling factor (default: 2.0)

        Returns:
            Upscaled latent tensor [B, C, H*scale, W*scale]
        """
        # Ensure input is 4D
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got {x.shape}")

        # Encode features
        features = self.encoder(x)

        # Calculate target size
        target_size = (int(x.shape[2] * scale), int(x.shape[3] * scale))

        # Adaptive upsampling based on scale factor
        if abs(scale - 2.0) > 0.01:  # Use epsilon for float comparison
            # For non-2x scales, use interpolation + convolution
            features = F.interpolate(features, size=target_size, mode='bilinear', align_corners=False)
            # Apply the convolution layers without the transpose conv
            for i, layer in enumerate(self.upsampler):
                if i == 0 and isinstance(layer, nn.ConvTranspose2d):
                    continue  # Skip transpose conv for non-2x scales
                features = layer(features)
            upsampled = features
        else:
            # For 2x scale, use transpose convolution
            upsampled = self.upsampler(features)

        # Skip connection with interpolation
        skip = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        skip = self.skip_conv(skip)

        # Combine upsampled features with skip connection
        return upsampled + skip
    
    @staticmethod
    def load_model(weight_path: str, device: torch.device, dtype: torch.dtype,
                   in_channels: int = 4, out_channels: int = 4, hidden_dim: int = 128) -> 'LatentResizer':
        """Load a trained model from checkpoint."""
        try:
            # Try to load our custom architecture first
            model = LatentResizer(in_channels=in_channels, out_channels=out_channels, hidden_dim=hidden_dim)
            state_dict = torch.load(weight_path, map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device=device, dtype=dtype)
            model.eval()
            return model
        except Exception as e:
            # If that fails, try to load the original architecture
            try:
                from .original_resizer import OriginalLatentResizer
                model = OriginalLatentResizer(in_channels=in_channels, out_channels=out_channels)
                state_dict = torch.load(weight_path, map_location=device)
                model.load_state_dict(state_dict)
                model = model.to(device=device, dtype=dtype)
                model.eval()
                return model
            except Exception as e2:
                raise Exception(f"Failed to load model with both architectures. Original error: {e}, Fallback error: {e2}")
    
    def save_model(self, save_path: str):
        """Save model state dict."""
        torch.save(self.state_dict(), save_path)


class WanLatentResizer(nn.Module):
    """
    Specialized resizer for Wan2.2 latents with 16 channels.
    This matches our trained model architecture exactly.
    """

    def __init__(self, in_channels: int = 16, out_channels: int = 16, hidden_dim: int = 256):
        super().__init__()

        # Encoder: 16 -> 64 -> 128 -> 256 (matches our trained model)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),  # encoder.0
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),          # encoder.2
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),         # encoder.4
            nn.ReLU(inplace=True)
        )

        # Upsampler: 256 -> 128 -> 64 -> 16 (matches our trained model)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # upsampler.0
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),                     # upsampler.2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)             # upsampler.4
        )

        # Skip connection
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, scale=2.0):
        # Store input for skip connection
        skip = self.skip_conv(x)

        # Encode
        features = self.encoder(x)

        # Upsample
        upsampled = self.upsampler(features)

        # Resize skip connection to match upsampled size
        if upsampled.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=upsampled.shape[2:], mode='bilinear', align_corners=False)

        # Add skip connection
        output = upsampled + skip

        # Ensure output matches expected scale
        if scale != 2.0:
            target_h = int(x.shape[2] * scale)
            target_w = int(x.shape[3] * scale)
            output = F.interpolate(output, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return output


class FluxLatentResizer(LatentResizer):
    """Specialized resizer for Flux latents with 16 channels."""
    
    def __init__(self, in_channels: int = 16, out_channels: int = 16, hidden_dim: int = 256):
        super().__init__(in_channels, out_channels, hidden_dim)


# Model configurations for different architectures
MODEL_CONFIGS = {
    "SD 1.5": {
        "in_channels": 4,
        "out_channels": 4,
        "hidden_dim": 128,
        "scale_factor": 0.13025,
        "class": LatentResizer,
        "use_original": True  # Use original UNet architecture
    },
    "SDXL": {
        "in_channels": 4,
        "out_channels": 4,
        "hidden_dim": 128,
        "scale_factor": 0.13025,
        "class": LatentResizer,
        "use_original": True  # Use original UNet architecture
    },
    "Flux": {
        "in_channels": 16,
        "out_channels": 16,
        "hidden_dim": 256,
        "scale_factor": 0.3611,
        "class": FluxLatentResizer,
        "use_original": True  # Use original UNet architecture
    },
    "Wan2.2": {
        "in_channels": 16,
        "out_channels": 16,
        "hidden_dim": 256,
        "scale_factor": 0.3604,  # From our training
        "class": WanLatentResizer,
        "use_original": False  # Use our custom architecture
    }
}


def create_resizer(model_type: str, **kwargs) -> LatentResizer:
    """Factory function to create appropriate resizer."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    resizer_class = config["class"]
    
    # Override config with any provided kwargs
    params = {
        "in_channels": config["in_channels"],
        "out_channels": config["out_channels"],
        "hidden_dim": config["hidden_dim"]
    }
    params.update(kwargs)
    
    return resizer_class(**params)


def get_scale_factor(model_type: str) -> float:
    """Get the scale factor for a specific model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODEL_CONFIGS[model_type]["scale_factor"]


def get_model_config(model_type: str) -> dict:
    """Get the full configuration for a specific model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODEL_CONFIGS[model_type].copy()
