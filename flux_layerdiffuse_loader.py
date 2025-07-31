"""
Flux LayerDiffuse Model Loader Node for ComfyUI
Loads TransparentVAE and LoRA weights for transparent image generation
"""

import os
import torch

# Handle relative imports for both ComfyUI and standalone testing
try:
    from .transparent_vae import TransparentVAE
except ImportError:
    from transparent_vae import TransparentVAE

# ComfyUI imports - only available when running in ComfyUI
try:
    import folder_paths
    import comfy.model_management as model_management
    import comfy.utils
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    # Fallback for testing outside ComfyUI
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_management = MockModelManagement()


class FluxLayerDiffuseVAELoader:
    """
    Node to create TransparentVAE from standard ComfyUI VAE + TransparentVAE weights
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "transparent_vae_weights": ("STRING", {
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
    FUNCTION = "load_transparent_vae"
    CATEGORY = "FluxLayerDiffuse"
    
    def load_transparent_vae(self, vae, transparent_vae_weights, alpha, latent_channels, dtype):
        """
        Create TransparentVAE from ComfyUI VAE and weights file
        """

        # Convert dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map[dtype]

        # Find TransparentVAE weights file in ComfyUI model directories
        if COMFY_AVAILABLE:
            import folder_paths

            # Look for the file in various model directories
            possible_paths = [
                os.path.join(folder_paths.models_dir, "vae", transparent_vae_weights),
                os.path.join(folder_paths.models_dir, "checkpoints", transparent_vae_weights),
                os.path.join(folder_paths.models_dir, transparent_vae_weights),
                transparent_vae_weights  # Direct path
            ]
        else:
            possible_paths = [transparent_vae_weights]

        transparent_vae_path = None
        for path in possible_paths:
            if os.path.exists(path):
                transparent_vae_path = path
                break

        if transparent_vae_path is None:
            raise FileNotFoundError(f"TransparentVAE weights not found. Tried: {possible_paths}")

        # Load TransparentVAE
        try:
            # Create TransparentVAE with the base VAE
            transparent_vae = TransparentVAE(
                sd_vae=vae.first_stage_model,
                dtype=torch_dtype,
                alpha=alpha,
                latent_c=latent_channels
            )

            # Load the TransparentVAE specific weights
            print(f"Loading TransparentVAE weights from: {transparent_vae_path}")
            state_dict = torch.load(transparent_vae_path, map_location="cpu")

            # Only load the decoder and encoder parts that are specific to TransparentVAE
            # Filter out incompatible keys
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('decoder.') or key.startswith('encoder.'):
                    # These are the TransparentVAE specific parts
                    filtered_state_dict[key] = value

            # Load only the compatible parts
            missing_keys, unexpected_keys = transparent_vae.load_state_dict(filtered_state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys (expected): {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")

            # Move to appropriate device
            device = model_management.get_torch_device()
            transparent_vae.to(device)

            print(f"✓ Loaded TransparentVAE from: {transparent_vae_path}")

        except Exception as e:
            print(f"Error loading TransparentVAE: {str(e)}")
            raise RuntimeError(f"Failed to load TransparentVAE: {str(e)}")

        return (transparent_vae,)


class FluxLayerDiffuseModelManager:
    """
    Utility class to manage model paths and downloads
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "action": (["check_models", "download_info"], {
                    "default": "check_models"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "manage_models"
    CATEGORY = "FluxLayerDiffuse"
    
    def manage_models(self, action):
        """
        Check model availability or provide download information
        """
        
        model_paths = {
            "TransparentVAE": "models/TransparentVAE.pth",
            "LayerLoRA": "models/layerlora.safetensors"
        }
        
        if action == "check_models":
            status_lines = ["=== Flux LayerDiffuse Model Status ==="]
            all_found = True
            
            for model_name, path in model_paths.items():
                if os.path.exists(path):
                    status_lines.append(f"✓ {model_name}: Found at {path}")
                else:
                    status_lines.append(f"✗ {model_name}: Missing at {path}")
                    all_found = False
            
            if all_found:
                status_lines.append("\n✓ All models are available!")
            else:
                status_lines.append("\n⚠ Some models are missing. Use 'download_info' action for instructions.")
            
            return ("\n".join(status_lines),)
        
        elif action == "download_info":
            info = """
=== Flux LayerDiffuse Model Download Instructions ===

1. Download models from HuggingFace:
   huggingface-cli download --resume-download --local-dir ./models RedAIGC/Flux-version-LayerDiffuse

2. Or manually download:
   - TransparentVAE.pth: https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/blob/main/TransparentVAE.pth
   - layerlora.safetensors: https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse/blob/main/layerlora.safetensors

3. Place files in:
   - models/TransparentVAE.pth
   - models/layerlora.safetensors

4. Make sure you have the required dependencies installed:
   pip install diffusers==0.32.2 safetensors transformers peft

Note: You also need a Flux.1-dev model for the pipeline.
"""
            return (info,)
        
        return ("Unknown action",)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseVAELoader": FluxLayerDiffuseVAELoader,
    "FluxLayerDiffuseModelManager": FluxLayerDiffuseModelManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseVAELoader": "Flux LayerDiffuse VAE Loader",
    "FluxLayerDiffuseModelManager": "Flux LayerDiffuse Model Manager",
}
