"""
Universal Latent Upscaler
Neural network-based latent upscaling for high-quality results.

This node uses a neural network trained on authentic VAE latents
to upscale latent representations from 32x32 to 64x64.
"""

import torch
import torch.nn as nn
import os
import folder_paths
import comfy.model_management as model_management

class ResidualBlock(nn.Module):
    """Residual Block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class AdvancedLatentUpscaler(nn.Module):
    """
    🔥 ADVANCED LATENT UPSCALER V2.0 🔥
    Trained with DENRAKEIW SUPERHERO SYSTEM!
    Input: [4, 32, 32] -> Output: [4, 64, 64]
    """
    def __init__(self, input_channels=4, output_channels=4, num_residual_blocks=8):
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, 64, 3, padding=1)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_residual_blocks)
        ])

        # Upsampling layers
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),  # 64 channels -> 64 channels, 2x upscale
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Initial features
        features = self.initial_conv(x)

        # Residual processing
        for block in self.residual_blocks:
            features = block(features)

        # Upsampling
        upsampled = self.upsample1(features)

        # Final output
        output = self.final_conv(upsampled)

        return output

# Legacy WAN VAE Upscaler (16 channels)
class WanNNLatentUpscaler(nn.Module):
    """
    Legacy Universal Latent Upscaler for WAN VAE (16 channels)
    """
    def __init__(self):
        super().__init__()

        # Input: [16, 32, 32] -> Output: [16, 64, 64]

        # Encoder für bessere Feature-Extraktion
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Upsampling-Pfad
        self.upsample = nn.Sequential(
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output-Layer
        self.output = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Tanh()  # Für VAE-Latents geeignet
        )

        # Residual Connection für bessere Gradients
        self.residual_proj = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)

    def forward(self, x):
        # x: [B, 16, 32, 32]

        # Residual connection
        residual = self.residual_proj(x)  # [B, 16, 64, 64]

        # Main path
        features = self.encoder(x)  # [B, 64, 32, 32]
        upsampled = self.upsample(features)  # [B, 32, 64, 64]
        output = self.output(upsampled)  # [B, 16, 64, 64]

        # Combine with residual
        result = output + residual

        return result

class WanNNLatentUpscalerNode:
    """
    🔥 ADVANCED LATENT UPSCALER V2.0 🔥

    Automatically detects latent type and uses the appropriate model:
    - 4 Channels: Advanced Model (trained by DENRAKEIW SUPERHERO)
    - 16 Channels: Legacy WAN VAE Model

    Upscales latent representations using neural networks trained
    on authentic VAE latents for high-quality results.
    """

    def __init__(self):
        self.advanced_model = None  # 4-channel model
        self.legacy_model = None    # 16-channel model
        self.device = model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "model_type": (["auto", "advanced", "legacy"], {
                    "default": "auto"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("upscaled_latent",)
    FUNCTION = "upscale_latent"
    CATEGORY = "denrakeiw/latent"
    DESCRIPTION = "🔥 ADVANCED LATENT UPSCALER V2.0 - Trained by DENRAKEIW SUPERHERO! Auto-detects 4ch/16ch latents and uses appropriate model for maximum quality!"
    
    def load_advanced_model(self):
        """Load the ADVANCED 4-channel model (DENRAKEIW SUPERHERO TRAINED)"""
        if self.advanced_model is not None:
            return

        # Try multiple possible paths for the new model
        possible_paths = [
            os.path.join("custom_nodes", "denrakeiw_nodes", "models", "best_model.pth"),
            os.path.join("models", "best_model.pth"),
            os.path.join("custom_nodes", "denrakeiw_nodes", "models", "final_advanced_upscaler.pth"),
            os.path.join("models", "final_advanced_upscaler.pth"),
        ]

        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                f"🔥 ADVANCED LATENT UPSCALER MODEL NOT FOUND!\n"
                f"Searched paths:\n" + "\n".join(f"  - {p}" for p in possible_paths) +
                f"\n\n💪 Please run the DENRAKEIW SUPERHERO training script first!"
            )

        try:
            print(f"🔥 Loading ADVANCED Latent Upscaler: {model_path}")

            # Create ADVANCED model
            self.advanced_model = AdvancedLatentUpscaler(
                input_channels=4,
                output_channels=4,
                num_residual_blocks=8
            )

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                val_loss = checkpoint.get('val_loss', checkpoint.get('loss', 'unknown'))
                print(f"✅ Loaded ADVANCED model from epoch {epoch} with val_loss {val_loss}")
            else:
                state_dict = checkpoint
                print(f"✅ Loaded ADVANCED model state dict")

            # Load weights
            self.advanced_model.load_state_dict(state_dict)
            self.advanced_model.to(self.device)
            self.advanced_model.eval()

            print(f"🚀 ADVANCED Latent Upscaler loaded successfully!")
            print(f"💪 Device: {self.device}")
            print(f"🔥 DENRAKEIW SUPERHERO MODEL READY!")

        except Exception as e:
            raise RuntimeError(f"🔥 ADVANCED Latent Upscaler loading failed!\nError: {e}")

    def load_legacy_model(self):
        """Load the LEGACY 16-channel WAN VAE model"""
        if self.legacy_model is not None:
            return

        # Try multiple model paths for WAN VAE
        model_paths = [
            "models/real_wan_vae_upscaler_best.pth",
            "models/simple_wan_vae_upscaler.pth",
            "models/wan_nn_latent_best.pth",
            "custom_nodes/denrakeiw_nodes/models/simple_wan_vae_upscaler.pth"
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print(f"⚠️ WAN VAE model not found in any of these paths:")
            for path in model_paths:
                print(f"   - {path}")
            print(f"   Using ADVANCED model for all inputs")
            return

        try:
            print(f"📦 Loading WAN VAE Upscaler: {model_path}")

            # Determine model type based on filename
            if "real_wan_vae_upscaler" in model_path:
                # Load Real WAN VAE Model (NEW!)
                from train_real_wan_vae_upscaler import RealWanVAEUpscaler
                self.legacy_model = RealWanVAEUpscaler(input_channels=16, output_channels=16)
                print(f"✅ Created REAL WAN VAE model (16-Channel)")
            elif "simple_wan_vae" in model_path:
                # Load Simple WAN VAE Model
                from simple_wan_vae_training import SimpleWanVAEUpscaler
                self.legacy_model = SimpleWanVAEUpscaler(input_channels=16, output_channels=16)
                print(f"✅ Created Simple WAN VAE model")
            else:
                # Load Legacy model
                self.legacy_model = WanNNLatentUpscaler()
                print(f"✅ Created Legacy WAN VAE model")

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('loss', checkpoint.get('val_loss', 'unknown'))
                print(f"✅ Loaded WAN VAE model from epoch {epoch} with loss {loss}")
            else:
                state_dict = checkpoint
                print(f"✅ Loaded WAN VAE model state dict")

            # Load weights
            self.legacy_model.load_state_dict(state_dict)
            self.legacy_model.to(self.device)
            self.legacy_model.eval()

            print(f"📦 WAN VAE Upscaler loaded successfully!")
            if "real_wan_vae_upscaler" in model_path:
                print(f"🔥 DENRAKEIW SUPERHERO REAL WAN VAE MODEL READY!")
                print(f"🌍 WELTHERSCHAFT THROUGH 16-CHANNEL LATENT UPSCALING!")
            else:
                print(f"🔥 DENRAKEIW SUPERHERO WAN VAE MODEL READY!")

        except Exception as e:
            print(f"⚠️ WAN VAE model loading failed: {e}")
            print(f"   Continuing with ADVANCED model only")
            import traceback
            traceback.print_exc()
    
    def upscale_latent(self, latent, strength=1.0, model_type="auto"):
        """
        🔥 ADVANCED LATENT UPSCALING V2.0 🔥
        Automatically detects latent type and uses appropriate model
        """

        # Get latent samples
        latent_samples = latent["samples"]

        # Check input dimensions
        batch_size, channels, height, width = latent_samples.shape

        print(f"🔥 ADVANCED LATENT UPSCALING V2.0:")
        print(f"   Input: {latent_samples.shape}")
        print(f"   Strength: {strength}")
        print(f"   Model Type: {model_type}")

        # Determine which model to use
        use_advanced = False
        use_legacy = False

        if model_type == "auto":
            if channels == 4:
                use_advanced = True
                print(f"🚀 AUTO-DETECTED: 4 channels -> Using ADVANCED model")
            elif channels == 16:
                use_legacy = True
                print(f"📦 AUTO-DETECTED: 16 channels -> Using LEGACY model")
            else:
                # Try to adapt
                print(f"⚠️ Unusual channel count: {channels}")
                print(f"   Trying ADVANCED model with channel adaptation...")
                use_advanced = True
        elif model_type == "advanced":
            use_advanced = True
            print(f"🔥 FORCED: Using ADVANCED model")
        elif model_type == "legacy":
            use_legacy = True
            print(f"📦 FORCED: Using LEGACY model")

        # Load appropriate model
        if use_advanced:
            self.load_advanced_model()
            model = self.advanced_model
            expected_channels = 4
            model_name = "ADVANCED"
        elif use_legacy:
            self.load_legacy_model()
            if self.legacy_model is None:
                print(f"⚠️ LEGACY model not available, falling back to ADVANCED")
                self.load_advanced_model()
                model = self.advanced_model
                expected_channels = 4
                model_name = "ADVANCED (fallback)"
            else:
                model = self.legacy_model
                expected_channels = 16
                model_name = "LEGACY"

        # Channel adaptation if needed
        input_tensor = latent_samples.to(self.device)

        if channels != expected_channels:
            print(f"🔧 CHANNEL ADAPTATION: {channels} -> {expected_channels}")
            if channels < expected_channels:
                # Pad channels
                padding = expected_channels - channels
                input_tensor = torch.cat([
                    input_tensor,
                    torch.zeros(batch_size, padding, height, width, device=self.device)
                ], dim=1)
                print(f"   Padded {padding} channels")
            elif channels > expected_channels:
                # Truncate channels
                input_tensor = input_tensor[:, :expected_channels, :, :]
                print(f"   Truncated to {expected_channels} channels")

        # Size validation
        if height != 32 or width != 32:
            print(f"⚠️ Input size {height}x{width} != 32x32")
            print(f"   This model was trained for 32x32 -> 64x64 upscaling")
            print(f"   Results may be suboptimal!")

        # Fix dtype mismatch - ensure model and input have same dtype
        model_dtype = next(model.parameters()).dtype
        if input_tensor.dtype != model_dtype:
            print(f"🔧 DTYPE FIX: Converting {input_tensor.dtype} -> {model_dtype}")
            input_tensor = input_tensor.to(model_dtype)

        # Upscale with model
        with torch.no_grad():
            try:
                # Run through neural network
                upscaled = model(input_tensor)

                # Restore original channel count if adapted
                if upscaled.shape[1] != channels:
                    if upscaled.shape[1] > channels:
                        upscaled = upscaled[:, :channels, :, :]
                        print(f"🔧 Restored to {channels} channels")
                    else:
                        # This shouldn't happen, but handle it
                        padding = channels - upscaled.shape[1]
                        upscaled = torch.cat([
                            upscaled,
                            torch.zeros(batch_size, padding, upscaled.shape[2], upscaled.shape[3], device=self.device)
                        ], dim=1)
                        print(f"🔧 Padded output to {channels} channels")

                # Apply strength
                if strength != 1.0:
                    # Blend with simple upsampling
                    simple_upscale = torch.nn.functional.interpolate(
                        latent_samples.to(self.device),
                        size=(upscaled.shape[2], upscaled.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                    upscaled = simple_upscale + strength * (upscaled - simple_upscale)

                # Move back to CPU
                upscaled = upscaled.cpu()

                print(f"🎉 {model_name} Latent Upscaling successful!")
                print(f"   Output: {upscaled.shape}")
                print(f"🔥 DENRAKEIW SUPERHERO POWER ACTIVATED!")

                # Return new latent
                return ({"samples": upscaled},)

            except Exception as e:
                print(f"❌ {model_name} Latent Upscaling failed: {e}")
                raise RuntimeError(f"🔥 ADVANCED Latent Upscaling error!\n{e}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "WanNNLatentUpscaler": WanNNLatentUpscalerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanNNLatentUpscaler": "🔥 Advanced Latent Upscaler V2.0"
}
