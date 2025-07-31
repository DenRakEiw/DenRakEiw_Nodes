"""
Flux LayerDiffuse Image-to-Image Node for ComfyUI
Generates transparent images from input images using Flux LayerDiffuse
"""

import torch
import numpy as np
from PIL import Image
from diffusers import FluxImg2ImgPipeline
import cv2

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


class FluxLayerDiffuseI2I:
    """
    Node for generating transparent images from input images using Flux LayerDiffuse
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
                "input_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "a handsome man with curly hair, high quality",
                    "multiline": True
                }),
                "strength": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
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
    FUNCTION = "generate_transparent_i2i"
    CATEGORY = "FluxLayerDiffuse"
    
    def generate_transparent_i2i(self, flux_model_path, transparent_vae, lora_path, input_image,
                               prompt, strength, steps, guidance_scale, seed, use_augmentation):
        """
        Generate transparent image from input image
        """
        
        device = model_management.get_torch_device()
        
        try:
            # Load Flux Img2Img pipeline
            print(f"Loading Flux Img2Img pipeline from: {flux_model_path}")
            pipe = FluxImg2ImgPipeline.from_pretrained(
                flux_model_path, 
                torch_dtype=torch.bfloat16
            ).to(device)
            
            # Load LoRA weights
            print(f"Loading LoRA weights from: {lora_path}")
            pipe.load_lora_weights(lora_path)
            
            # Move TransparentVAE to device
            transparent_vae.to(device)
            
            # Process input image
            if input_image.shape[0] > 1:
                print("Warning: Multiple input images detected, using the first one")
                input_image = input_image[0:1]
            
            # Convert from ComfyUI format (BHWC) to PIL Image
            input_image_np = input_image[0].cpu().numpy()
            
            # Handle different input formats
            if input_image_np.shape[-1] == 4:  # RGBA
                # Convert RGBA to RGB for pipeline, preserve alpha for later
                rgb_image = input_image_np[:, :, :3]
                alpha_channel = input_image_np[:, :, 3:4]
                has_alpha = True
            else:  # RGB
                rgb_image = input_image_np
                has_alpha = False
            
            # Convert to PIL Image
            rgb_image_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
            
            print(f"Processing image: {rgb_image_pil.size}")
            print(f"Generating transparent image with prompt: '{prompt}'")
            
            # Generate using Flux Img2Img pipeline
            generator = torch.Generator(device).manual_seed(seed)
            
            result = pipe(
                prompt=prompt,
                image=rgb_image_pil,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent"
            )
            
            latents = result.images
            
            # Unpack latents
            latents = pipe._unpack_latents(latents, rgb_image_pil.height, rgb_image_pil.width, pipe.vae_scale_factor)
            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            
            # Decode using TransparentVAE
            with torch.no_grad():
                original_x, transparent_image = transparent_vae.decode(latents, aug=use_augmentation)
            
            # Process output
            transparent_image = transparent_image.clamp(0, 1)
            transparent_image = transparent_image.permute(0, 2, 3, 1)  # BCHW -> BHWC
            
            # Convert to numpy for ComfyUI
            transparent_image_np = transparent_image.float().cpu().numpy()
            
            print(f"✓ Generated transparent image: {transparent_image_np.shape}")
            
            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return (transparent_image_np,)
            
        except Exception as e:
            print(f"Error in transparent I2I generation: {str(e)}")
            raise RuntimeError(f"Failed to generate transparent image: {str(e)}")


class FluxLayerDiffuseI2IAdvanced:
    """
    Advanced I2I version with mask support and more options
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
                "input_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "a handsome man with curly hair, high quality",
                    "multiline": True
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "strength": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
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
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("transparent_image", "original_image")
    FUNCTION = "generate_transparent_i2i_advanced"
    CATEGORY = "FluxLayerDiffuse"
    
    def generate_transparent_i2i_advanced(self, flux_model_path, transparent_vae, lora_path, 
                                        input_image, prompt, negative_prompt, strength, steps, 
                                        guidance_scale, seed, use_augmentation, mask=None):
        """
        Advanced transparent I2I generation with mask support
        """
        
        device = model_management.get_torch_device()
        
        try:
            # Load Flux Img2Img pipeline
            print(f"Loading Flux Img2Img pipeline from: {flux_model_path}")
            pipe = FluxImg2ImgPipeline.from_pretrained(
                flux_model_path, 
                torch_dtype=torch.bfloat16
            ).to(device)
            
            # Load LoRA weights
            print(f"Loading LoRA weights from: {lora_path}")
            pipe.load_lora_weights(lora_path)
            
            # Move TransparentVAE to device
            transparent_vae.to(device)
            
            # Process input image
            if input_image.shape[0] > 1:
                print("Warning: Multiple input images detected, using the first one")
                input_image = input_image[0:1]
            
            # Convert from ComfyUI format (BHWC) to PIL Image
            input_image_np = input_image[0].cpu().numpy()
            
            # Handle different input formats
            if input_image_np.shape[-1] == 4:  # RGBA
                rgb_image = input_image_np[:, :, :3]
                alpha_channel = input_image_np[:, :, 3:4]
                has_alpha = True
            else:  # RGB
                rgb_image = input_image_np
                has_alpha = False
            
            # Convert to PIL Image
            rgb_image_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))
            
            # Process mask if provided
            mask_pil = None
            if mask is not None:
                mask_np = mask[0].cpu().numpy()  # Take first mask if batch
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                print(f"Using mask: {mask_pil.size}")
            
            print(f"Processing image: {rgb_image_pil.size}")
            print(f"Generating transparent image with prompt: '{prompt}'")
            
            # Generate using Flux Img2Img pipeline
            generator = torch.Generator(device).manual_seed(seed)
            
            # Prepare pipeline arguments
            pipe_args = {
                "prompt": prompt,
                "image": rgb_image_pil,
                "strength": strength,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
                "output_type": "latent"
            }
            
            if negative_prompt:
                pipe_args["negative_prompt"] = negative_prompt
            
            if mask_pil is not None:
                pipe_args["mask_image"] = mask_pil
            
            result = pipe(**pipe_args)
            latents = result.images
            
            # Unpack latents
            latents = pipe._unpack_latents(latents, rgb_image_pil.height, rgb_image_pil.width, pipe.vae_scale_factor)
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
            
            print(f"✓ Generated transparent I2I image: {transparent_images_np.shape}")
            
            # Clean up GPU memory
            del pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return (transparent_images_np, original_images_np)
            
        except Exception as e:
            print(f"Error in advanced transparent I2I generation: {str(e)}")
            raise RuntimeError(f"Failed to generate transparent image: {str(e)}")


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseI2I": FluxLayerDiffuseI2I,
    "FluxLayerDiffuseI2IAdvanced": FluxLayerDiffuseI2IAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseI2I": "Flux LayerDiffuse I2I",
    "FluxLayerDiffuseI2IAdvanced": "Flux LayerDiffuse I2I Advanced",
}
