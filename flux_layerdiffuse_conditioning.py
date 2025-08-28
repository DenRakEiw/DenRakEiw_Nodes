"""
Flux LayerDiffuse Conditioning Helpers
Handles CLIP encoding compatibility issues with Flux models
"""

import torch
import numpy as np

# ComfyUI imports
import comfy.model_management as model_management


class FluxLayerDiffuseConditioningFix:
    """
    Fix conditioning tensor dimensions for Flux LayerDiffuse
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "target_length": ("INT", {
                    "default": 256,
                    "min": 77,
                    "max": 512,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("fixed_conditioning",)
    FUNCTION = "fix_conditioning"
    CATEGORY = "denrakeiw/flux"

    def fix_conditioning(self, conditioning, target_length):
        """
        Fix conditioning tensor dimensions for Flux compatibility
        """

        try:
            fixed_conditioning = []

            for cond in conditioning:
                # Extract the conditioning tensor
                cond_tensor = cond[0]  # Shape should be [batch, seq_len, hidden_dim]
                cond_dict = cond[1].copy() if len(cond) > 1 else {}

                print(f"Original conditioning shape: {cond_tensor.shape}")

                # Check if we need to fix the dimensions
                if len(cond_tensor.shape) == 3:
                    batch_size, seq_len, hidden_dim = cond_tensor.shape

                    # Flux expects 4096 hidden dimensions
                    target_hidden_dim = 4096

                    # Fix sequence length
                    if seq_len != target_length:
                        if seq_len < target_length:
                            # Pad with zeros
                            padding = torch.zeros(batch_size, target_length - seq_len, hidden_dim,
                                                device=cond_tensor.device, dtype=cond_tensor.dtype)
                            cond_tensor = torch.cat([cond_tensor, padding], dim=1)
                        else:
                            # Truncate
                            cond_tensor = cond_tensor[:, :target_length, :]

                    # Fix hidden dimension
                    if hidden_dim != target_hidden_dim:
                        if hidden_dim < target_hidden_dim:
                            # Pad hidden dimension with zeros
                            padding = torch.zeros(batch_size, target_length, target_hidden_dim - hidden_dim,
                                                device=cond_tensor.device, dtype=cond_tensor.dtype)
                            cond_tensor = torch.cat([cond_tensor, padding], dim=2)
                        else:
                            # Truncate hidden dimension
                            cond_tensor = cond_tensor[:, :, :target_hidden_dim]

                    print(f"Fixed conditioning shape: {cond_tensor.shape}")

                # Create new conditioning tuple
                fixed_conditioning.append([cond_tensor, cond_dict])

            return (fixed_conditioning,)

        except Exception as e:
            print(f"Error in conditioning fix: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original conditioning if fix fails
            return (conditioning,)


class FluxLayerDiffuseEmptyConditioning:
    """
    Create empty conditioning for Flux LayerDiffuse (for negative prompts)
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "sequence_length": ("INT", {
                    "default": 256,
                    "min": 77,
                    "max": 512,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("empty_conditioning",)
    FUNCTION = "create_empty_conditioning"
    CATEGORY = "denrakeiw/flux"

    def create_empty_conditioning(self, clip, batch_size, sequence_length):
        """
        Create empty conditioning tensor for negative prompts
        """
        
        try:
            # Get CLIP model info
            device = model_management.get_torch_device()
            
            # Try to get hidden dimension from CLIP
            try:
                # This might vary depending on the CLIP model
                hidden_dim = 4096  # Common for Flux CLIP models
                
                # Create empty conditioning tensor
                empty_tensor = torch.zeros(batch_size, sequence_length, hidden_dim, 
                                         device=device, dtype=torch.float32)
                
                # Create conditioning in ComfyUI format
                conditioning = [[empty_tensor, {}]]
                
                print(f"Created empty conditioning: {empty_tensor.shape}")
                
                return (conditioning,)
                
            except Exception as e:
                print(f"Error creating empty conditioning: {e}")
                # Fallback: try to encode empty string
                tokens = clip.tokenize("")
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                return ([[cond, {"pooled_output": pooled}]],)
            
        except Exception as e:
            print(f"Error in empty conditioning creation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to create empty conditioning: {str(e)}")





# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseConditioningFix": FluxLayerDiffuseConditioningFix,
    "FluxLayerDiffuseEmptyConditioning": FluxLayerDiffuseEmptyConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseConditioningFix": "🔧 Flux LayerDiffuse Conditioning Fix",
    "FluxLayerDiffuseEmptyConditioning": "⭕ Flux LayerDiffuse Empty Conditioning",
}
