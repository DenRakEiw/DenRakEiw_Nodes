"""
Simplified Flux LayerDiffuse Nodes for ComfyUI
Uses standard ComfyUI model loading and sampling
"""

import torch
import numpy as np

# ComfyUI imports - only available when running in ComfyUI
try:
    import comfy.model_management as model_management
    import comfy.sample
    import comfy.samplers
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    # Fallback for testing outside ComfyUI
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_management = MockModelManagement()

# Handle relative imports for both ComfyUI and standalone testing
try:
    from .transparent_vae import TransparentVAE
except ImportError:
    from transparent_vae import TransparentVAE


class FluxLayerDiffuseSampler:
    """
    Simple sampler that uses ComfyUI's standard sampling with TransparentVAE decoding
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "transparent_vae": ("TRANSPARENT_VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 10000
                }),
                "cfg": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "ddim", "uni_pc", "uni_pc_bh2"],),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"],),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "use_augmentation": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "transparent_image")
    FUNCTION = "sample_transparent"
    CATEGORY = "FluxLayerDiffuse"

    def sample_transparent(self, model, transparent_vae, positive, negative, latent_image,
                         seed, steps, cfg, sampler_name, scheduler, denoise, use_augmentation):
        """
        Sample using ComfyUI's sampling system, then decode with TransparentVAE
        """

        try:
            # Set seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Use ComfyUI's sampling system
            import comfy.sample

            # Sample latents
            latent = comfy.sample.sample(
                model=model,
                noise=latent_image["samples"],
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                denoise=denoise,
                disable_noise=denoise < 1.0,
                start_step=0,
                last_step=steps,
                force_full_denoise=True,
                noise_mask=None,
                callback=None,
                disable_pbar=False,
                seed=seed
            )

            # Decode with TransparentVAE
            device = model_management.get_torch_device()
            transparent_vae.to(device)

            with torch.no_grad():
                original_x, transparent_image = transparent_vae.decode(latent, aug=use_augmentation)

            # Process output for ComfyUI
            transparent_image = transparent_image.clamp(0, 1)
            transparent_image = transparent_image.permute(0, 2, 3, 1)  # BCHW -> BHWC
            transparent_image_np = transparent_image.float().cpu().numpy()

            # Return both latent and decoded transparent image
            return ({"samples": latent}, transparent_image_np)

        except Exception as e:
            print(f"Error in FluxLayerDiffuseSampler: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Sampling failed: {str(e)}")


class FluxLayerDiffuseDecoder:
    """
    Decode latents using TransparentVAE
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

    def decode_transparent(self, transparent_vae, samples, use_augmentation):
        """
        Decode latents to transparent images
        """
        
        device = model_management.get_torch_device()
        transparent_vae.to(device)
        
        latent = samples["samples"]
        
        with torch.no_grad():
            original_x, transparent_image = transparent_vae.decode(latent, aug=use_augmentation)
        
        # Process output for ComfyUI
        transparent_image = transparent_image.clamp(0, 1)
        transparent_image = transparent_image.permute(0, 2, 3, 1)  # BCHW -> BHWC
        transparent_image_np = transparent_image.float().cpu().numpy()
        
        return (transparent_image_np,)


class EmptyLatentTransparent:
    """
    Create empty latent for transparent image generation
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 8192,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 16,
                    "max": 8192,
                    "step": 8
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4096
                })
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "FluxLayerDiffuse"

    def generate(self, width, height, batch_size=1):
        # Flux uses 16 latent channels and 8x downsampling
        latent = torch.zeros([batch_size, 16, height // 8, width // 8], dtype=torch.float32)
        return ({"samples": latent},)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseSampler": FluxLayerDiffuseSampler,
    "FluxLayerDiffuseDecoder": FluxLayerDiffuseDecoder,
    "EmptyLatentTransparent": EmptyLatentTransparent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseSampler": "🎨 Flux LayerDiffuse Sampler",
    "FluxLayerDiffuseDecoder": "🔍 Flux LayerDiffuse Decoder", 
    "EmptyLatentTransparent": "📋 Empty Latent (Transparent)",
}
