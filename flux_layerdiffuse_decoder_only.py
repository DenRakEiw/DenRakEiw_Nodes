"""
Simplified Flux LayerDiffuse Decoder - works with standard ComfyUI workflow
Use this with the standard KSampler node
"""

import torch
import numpy as np

# ComfyUI imports
import comfy.model_management as model_management

# TransparentVAE is loaded via the standalone loader


class FluxLayerDiffuseDecoderSimple:
    """
    Simple decoder that takes latents from any sampler and outputs transparent images
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transparent_vae": ("TRANSPARENT_VAE",),
                "samples": ("LATENT",),
                "use_augmentation": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("transparent_image",)
    FUNCTION = "decode_transparent"
    CATEGORY = "FluxLayerDiffuse"

    def decode_transparent(self, transparent_vae, samples, use_augmentation, **kwargs):
        """
        Decode latents to transparent images using TransparentVAE
        Note: **kwargs handles backward compatibility with old workflows that pass 'vae' parameter
        """

        try:
            # Check for old workflow compatibility
            if 'vae' in kwargs:
                print("⚠ Warning: 'vae' parameter is no longer needed in FluxLayerDiffuseDecoderSimple")
                print("   The VAE should be connected to FluxLayerDiffuseStandaloneLoader instead")

            device = model_management.get_torch_device()
            transparent_vae.to(device)

            latent = samples["samples"]
            print(f"Input latent shape: {latent.shape}")
            print(f"Latent device: {latent.device}, dtype: {latent.dtype}")
            print(f"TransparentVAE device: {next(transparent_vae.parameters()).device}")

            # Move latent to same device and dtype as TransparentVAE
            latent = latent.to(device=device, dtype=transparent_vae.dtype)

            print("Decoding with TransparentVAE (includes internal VAE decode)...")
            with torch.no_grad():
                # TransparentVAE.decode() does the VAE decoding internally
                original_x, transparent_image = transparent_vae.decode(latent, aug=use_augmentation)

            # Process output for ComfyUI
            transparent_image = transparent_image.clamp(0, 1)
            transparent_image = transparent_image.permute(0, 2, 3, 1)  # BCHW -> BHWC
            transparent_image_np = transparent_image.float().cpu().numpy()

            print(f"✓ Decoded transparent image: {transparent_image_np.shape}")

            return (transparent_image_np,)

        except Exception as e:
            print(f"Error in FluxLayerDiffuseDecoderSimple: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Decoding failed: {str(e)}")





# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseDecoderSimple": FluxLayerDiffuseDecoderSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseDecoderSimple": "🔍 Flux LayerDiffuse Decoder (Simple)",
}
