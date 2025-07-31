"""
Flux LayerDiffuse Text-to-Image Node for ComfyUI
Generates transparent images from text prompts using Flux LayerDiffuse
"""

import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline

# ComfyUI imports - only available when running in ComfyUI
try:
    import comfy.model_management as model_management
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    # Fallback for testing outside ComfyUI
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_management = MockModelManagement()


class FluxLayerDiffuseT2I:
    """
    Node for generating transparent images from text using Flux LayerDiffuse
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "transparent_vae": ("TRANSPARENT_VAE",),
                "conditioning": ("CONDITIONING",),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 11111,
                    "min": 0,
                    "max": 2**32 - 1
                }),
                "use_augmentation": ("BOOLEAN", {
                    "default": True
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("transparent_image",)
    FUNCTION = "generate_transparent_image"
    CATEGORY = "FluxLayerDiffuse"
    
    def generate_transparent_image(self, model, transparent_vae, conditioning,
                                 width, height, steps, guidance_scale, seed, use_augmentation):
        """
        Generate transparent image using ComfyUI model and conditioning
        """

        device = model_management.get_torch_device()

        try:
            # Move TransparentVAE to device
            transparent_vae.to(device)

            print(f"Generating transparent image with ComfyUI model")

            # Use ComfyUI's sampling system
            import comfy.sample
            import comfy.samplers

            # Create noise
            noise = torch.randn((1, 16, height // 8, width // 8), device=device, dtype=torch.bfloat16)

            # Set seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Sample using ComfyUI's sampling system
            sampler = comfy.samplers.KSampler(model, steps=steps, device=device)

            # Generate latents
            latents = sampler.sample(
                noise=noise,
                positive=conditioning,
                negative=None,  # No negative conditioning for now
                cfg=guidance_scale,
                denoise=1.0,
                disable_noise=False,
                start_step=0,
                last_step=steps,
                force_full_denoise=True
            )

            # Decode using TransparentVAE
            with torch.no_grad():
                original_x, transparent_image = transparent_vae.decode(latents, aug=use_augmentation)

            # Process output
            transparent_image = transparent_image.clamp(0, 1)
            transparent_image = transparent_image.permute(0, 2, 3, 1)  # BCHW -> BHWC

            # Convert to numpy for ComfyUI
            transparent_image_np = transparent_image.float().cpu().numpy()

            # Ensure we have RGBA channels
            if transparent_image_np.shape[-1] == 4:
                # Already RGBA
                result_image = transparent_image_np
            else:
                # Add alpha channel if missing
                alpha = torch.ones_like(transparent_image[:, :, :, :1])
                transparent_image_with_alpha = torch.cat([transparent_image, alpha], dim=-1)
                result_image = transparent_image_with_alpha.float().cpu().numpy()

            print(f"✓ Generated transparent image: {result_image.shape}")

            return (result_image,)

        except Exception as e:
            print(f"Error in transparent image generation: {str(e)}")
            raise RuntimeError(f"Failed to generate transparent image: {str(e)}")


class FluxLayerDiffuseT2IAdvanced:
    """
    Advanced version with more control options
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flux_model_path": ("STRING", {
                    "default": "black-forest-labs/FLUX.1-dev",
                    "multiline": False
                }),
                "transparent_vae": ("TRANSPARENT_VAE",),
                "lora_path": ("STRING", {
                    "default": "models/layerlora.safetensors",
                    "multiline": False
                }),
                "prompt": ("STRING", {
                    "default": "glass bottle, high quality",
                    "multiline": True
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 64
                }),
                "steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": 11111,
                    "min": 0,
                    "max": 2**32 - 1
                }),
                "use_augmentation": ("BOOLEAN", {
                    "default": True
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("transparent_image", "original_image")
    FUNCTION = "generate_transparent_image_advanced"
    CATEGORY = "FluxLayerDiffuse"
    
    def generate_transparent_image_advanced(self, flux_model_path, transparent_vae, lora_path, 
                                          prompt, negative_prompt, width, height, steps, 
                                          guidance_scale, seed, use_augmentation, batch_size):
        """
        Advanced transparent image generation with more options
        """
        
        device = model_management.get_torch_device()
        
        try:
            # Load Flux pipeline
            print(f"Loading Flux pipeline from: {flux_model_path}")
            pipe = FluxPipeline.from_pretrained(
                flux_model_path, 
                torch_dtype=torch.bfloat16
            ).to(device)
            
            # Load LoRA weights
            print(f"Loading LoRA weights from: {lora_path}")
            pipe.load_lora_weights(lora_path)
            
            # Move TransparentVAE to device
            transparent_vae.to(device)
            
            print(f"Generating {batch_size} transparent image(s) with prompt: '{prompt}'")
            
            # Generate latents using Flux pipeline
            generator = torch.Generator(device).manual_seed(seed)
            
            # Prepare prompts for batch
            prompts = [prompt] * batch_size
            negative_prompts = [negative_prompt] * batch_size if negative_prompt else None
            
            latents = pipe(
                prompt=prompts,
                negative_prompt=negative_prompts,
                height=height,
                width=width,
                num_inference_steps=steps,
                output_type="latent",
                generator=generator,
                guidance_scale=guidance_scale,
            ).images
            
            # Unpack latents
            latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            
            # Decode using TransparentVAE
            with torch.no_grad():
                original_images, transparent_images = transparent_vae.decode(latents, aug=use_augmentation)
            
            # Process transparent output
            transparent_images = transparent_images.clamp(0, 1)
            transparent_images = transparent_images.permute(0, 2, 3, 1)  # BCHW -> BHWC
            transparent_images_np = transparent_images.float().cpu().numpy()
            
            # Process original output
            original_images = original_images.clamp(0, 1)
            original_images = original_images.permute(0, 2, 3, 1)  # BCHW -> BHWC
            original_images_np = original_images.float().cpu().numpy()
            
            print(f"✓ Generated {batch_size} transparent image(s): {transparent_images_np.shape}")
            
            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return (transparent_images_np, original_images_np)
            
        except Exception as e:
            print(f"Error in advanced transparent image generation: {str(e)}")
            raise RuntimeError(f"Failed to generate transparent image: {str(e)}")


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseT2I": FluxLayerDiffuseT2I,
    "FluxLayerDiffuseT2IAdvanced": FluxLayerDiffuseT2IAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseT2I": "Flux LayerDiffuse T2I",
    "FluxLayerDiffuseT2IAdvanced": "Flux LayerDiffuse T2I Advanced",
}
