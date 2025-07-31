"""
Standalone Flux LayerDiffuse Loader that doesn't depend on ComfyUI VAE
Loads TransparentVAE directly from the checkpoint
"""

import torch
import os

# ComfyUI imports - only available when running in ComfyUI
try:
    import comfy.model_management as model_management
    import folder_paths
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


class FluxLayerDiffuseStandaloneLoader:
    """
    Load TransparentVAE directly from checkpoint without requiring ComfyUI VAE
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "transparent_vae_checkpoint": ("STRING", {
                    "default": "TransparentVAE.pth",
                    "multiline": False
                }),
                "alpha": ("FLOAT", {
                    "default": 300.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 1.0
                }),
                "latent_channels": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 32,
                    "step": 1
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16"
                }),
            }
        }
    
    RETURN_TYPES = ("TRANSPARENT_VAE",)
    RETURN_NAMES = ("transparent_vae",)
    FUNCTION = "load_standalone_transparent_vae"
    CATEGORY = "FluxLayerDiffuse"

    def load_standalone_transparent_vae(self, vae, transparent_vae_checkpoint, alpha, latent_channels, dtype):
        """
        Load TransparentVAE directly from checkpoint
        """
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        # Find TransparentVAE checkpoint file
        if COMFY_AVAILABLE:
            # Look for the file in various model directories
            possible_paths = [
                os.path.join(folder_paths.models_dir, "vae", transparent_vae_checkpoint),
                os.path.join(folder_paths.models_dir, "checkpoints", transparent_vae_checkpoint),
                os.path.join(folder_paths.models_dir, transparent_vae_checkpoint),
                os.path.join("models", "vae", transparent_vae_checkpoint),
                os.path.join("models", transparent_vae_checkpoint),
                transparent_vae_checkpoint  # Direct path
            ]
        else:
            possible_paths = [
                os.path.join("models", "vae", transparent_vae_checkpoint),
                os.path.join("models", transparent_vae_checkpoint),
                transparent_vae_checkpoint
            ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"TransparentVAE checkpoint not found. Tried: {possible_paths}")
        
        try:
            print(f"Loading TransparentVAE checkpoint from: {checkpoint_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Use the provided VAE directly
            device = model_management.get_torch_device()

            # ComfyUI VAE doesn't have .to() method, but first_stage_model does
            if hasattr(vae, 'first_stage_model'):
                actual_vae = vae.first_stage_model
            else:
                actual_vae = vae

            # Move actual VAE to device if it has .to() method
            if hasattr(actual_vae, 'to'):
                actual_vae.to(device)

            transparent_vae = TransparentVAE(
                sd_vae=actual_vae,  # Use the actual VAE model
                dtype=torch_dtype,
                alpha=alpha,
                latent_c=latent_channels
            )
            
            # Load the checkpoint weights
            if isinstance(checkpoint, dict):
                # If checkpoint contains state_dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load weights with strict=False to handle missing/extra keys
            missing_keys, unexpected_keys = transparent_vae.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in TransparentVAE: {len(missing_keys)}")
                # This is expected as we're not loading the base VAE weights
            
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
            
            # Move to appropriate device
            device = model_management.get_torch_device()
            transparent_vae.to(device)
            
            print(f"✓ Successfully loaded TransparentVAE from: {checkpoint_path}")
            print(f"✓ Device: {device}, dtype: {torch_dtype}")
            
            return (transparent_vae,)
            
        except Exception as e:
            print(f"Error loading TransparentVAE: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load TransparentVAE: {str(e)}")


class FluxLayerDiffuseInfo:
    """
    Display information about TransparentVAE and provide setup instructions
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "action": (["check_files", "setup_guide"], {
                    "default": "check_files"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "FluxLayerDiffuse"
    
    def get_info(self, action):
        """
        Provide information and setup guidance
        """
        
        if action == "check_files":
            info_lines = ["=== Flux LayerDiffuse File Check ==="]
            
            # Check for TransparentVAE
            if COMFY_AVAILABLE:
                vae_paths = [
                    os.path.join(folder_paths.models_dir, "vae", "TransparentVAE.pth"),
                    os.path.join(folder_paths.models_dir, "TransparentVAE.pth"),
                ]
                lora_paths = [
                    os.path.join(folder_paths.models_dir, "loras", "layerlora.safetensors"),
                ]
            else:
                vae_paths = ["models/TransparentVAE.pth"]
                lora_paths = ["models/layerlora.safetensors"]
            
            # Check TransparentVAE
            vae_found = False
            for path in vae_paths:
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    info_lines.append(f"✓ TransparentVAE.pth: Found ({size_mb:.1f} MB)")
                    info_lines.append(f"  Path: {path}")
                    vae_found = True
                    break
            
            if not vae_found:
                info_lines.append("✗ TransparentVAE.pth: Not found")
                info_lines.append("  Expected locations:")
                for path in vae_paths:
                    info_lines.append(f"    {path}")
            
            # Check LoRA
            lora_found = False
            for path in lora_paths:
                if os.path.exists(path):
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    info_lines.append(f"✓ layerlora.safetensors: Found ({size_mb:.1f} MB)")
                    info_lines.append(f"  Path: {path}")
                    lora_found = True
                    break
            
            if not lora_found:
                info_lines.append("✗ layerlora.safetensors: Not found")
                info_lines.append("  Expected locations:")
                for path in lora_paths:
                    info_lines.append(f"    {path}")
            
            if vae_found and lora_found:
                info_lines.append("\n🎉 All files found! You're ready to use Flux LayerDiffuse.")
            else:
                info_lines.append("\n⚠ Missing files. Use 'setup_guide' for download instructions.")
            
            return ("\n".join(info_lines),)
        
        elif action == "setup_guide":
            guide = """
=== Flux LayerDiffuse Setup Guide ===

1. Download Required Files:
   
   TransparentVAE.pth:
   https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/blob/main/TransparentVAE.pth
   → Place in: ComfyUI/models/vae/
   
   layerlora.safetensors:
   https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/blob/main/layerlora.safetensors
   → Place in: ComfyUI/models/loras/

2. Flux Model:
   Download any Flux.1-dev model
   → Place in: ComfyUI/models/diffusion_models/

3. Basic Workflow:
   - Load Diffusion Model (Flux)
   - Load LoRA (layerlora.safetensors, strength: 1.0)
   - Flux LayerDiffuse Standalone Loader
   - CLIP Text Encode ("glass bottle, transparent")
   - Empty Latent (1024x1024)
   - Flux LayerDiffuse Sampler
   - Save Image

4. Tips:
   - Use prompts like "glass bottle", "crystal object"
   - Avoid background descriptions
   - CFG 3.5-7.0 works well
   - Enable augmentation for better quality

Need help? Check the README.md in the node folder.
"""
            return (guide,)
        
        return ("Unknown action",)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseStandaloneLoader": FluxLayerDiffuseStandaloneLoader,
    "FluxLayerDiffuseInfo": FluxLayerDiffuseInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseStandaloneLoader": "🔧 Flux LayerDiffuse Standalone Loader",
    "FluxLayerDiffuseInfo": "ℹ️ Flux LayerDiffuse Info",
}
