"""
Simplified Flux LayerDiffuse Decoder - works with standard ComfyUI workflow
Use this with the standard KSampler node
"""

import torch
import numpy as np

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

# Handle relative imports for both ComfyUI and standalone testing
try:
    from .transparent_vae import TransparentVAE
except ImportError:
    from transparent_vae import TransparentVAE


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


class FluxLayerDiffuseWorkflowHelper:
    """
    Helper node that provides workflow guidance
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "workflow_type": (["basic_workflow", "advanced_workflow", "troubleshooting"], {
                    "default": "basic_workflow"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("instructions",)
    FUNCTION = "get_workflow_help"
    CATEGORY = "FluxLayerDiffuse"
    
    def get_workflow_help(self, workflow_type):
        """
        Provide workflow instructions
        """
        
        if workflow_type == "basic_workflow":
            instructions = """
=== Basic Flux LayerDiffuse Workflow ===

1. Load Diffusion Model
   → Load your Flux model from models/diffusion_models/

2. Load LoRA  
   → Load layerlora.safetensors with strength 1.0

3. CLIP Text Encode (Positive)
   → "glass bottle, transparent, high quality"

4. CLIP Text Encode (Negative) 
   → "" (empty or "blurry, low quality")

5. Empty Latent Image
   → Width: 1024, Height: 1024

6. KSampler
   → Connect: model, positive, negative, latent_image
   → Steps: 20-50, CFG: 3.5-7.0
   → Sampler: euler, Scheduler: normal

7. Flux LayerDiffuse Standalone Loader
   → transparent_vae_checkpoint: "TransparentVAE.pth"

8. Flux LayerDiffuse Decoder Simple
   → Connect: transparent_vae, samples from KSampler

9. Save Image
   → Connect transparent_image output

This workflow uses standard ComfyUI nodes + LayerDiffuse decoder!
"""
            
        elif workflow_type == "advanced_workflow":
            instructions = """
=== Advanced Flux LayerDiffuse Workflow ===

For better quality and control:

1. Use higher resolution (1152x896, 1344x768)
2. More sampling steps (30-50)
3. Enable augmentation in decoder
4. Try different CFG values (3.5-7.0)
5. Experiment with different samplers:
   - euler (fast, good quality)
   - dpmpp_2m (slower, higher quality)
   - ddim (deterministic)

Prompt tips:
✓ "crystal wine glass, elegant, transparent"
✓ "glass sculpture, artistic, clear"
✓ "transparent bottle, studio lighting"
✗ Avoid: backgrounds, "on white background"

Model settings:
- LoRA strength: 1.0 (full strength)
- Alpha in loader: 300.0 (default)
- Use augmentation: true (better quality)
"""
            
        elif workflow_type == "troubleshooting":
            instructions = """
=== Flux LayerDiffuse Troubleshooting ===

Common Issues:

1. "TransparentVAE.pth not found"
   → Check file is in ComfyUI/models/vae/
   → Use "Flux LayerDiffuse Info" node to verify

2. "Model not compatible" 
   → Ensure using Flux.1-dev model
   → Check LoRA is loaded correctly

3. Poor transparency quality
   → Increase CFG scale (try 5.0-7.0)
   → Enable augmentation in decoder
   → Use more sampling steps
   → Check prompt doesn't mention backgrounds

4. Memory errors
   → Reduce image size
   → Use dtype: float16 instead of bfloat16
   → Disable augmentation

5. Sampling errors
   → Use standard KSampler, not FluxLayerDiffuseSampler
   → Check model and LoRA are properly loaded

6. Wrong colors/artifacts
   → Try different samplers (euler, dpmpp_2m)
   → Adjust CFG scale
   → Check seed consistency
"""
        
        else:
            instructions = "Unknown workflow type"
        
        return (instructions,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseDecoderSimple": FluxLayerDiffuseDecoderSimple,
    "FluxLayerDiffuseWorkflowHelper": FluxLayerDiffuseWorkflowHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseDecoderSimple": "🔍 Flux LayerDiffuse Decoder (Simple)",
    "FluxLayerDiffuseWorkflowHelper": "📖 Flux LayerDiffuse Workflow Helper",
}
