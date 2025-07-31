"""
Flux LayerDiffuse Conditioning Helpers
Handles CLIP encoding compatibility issues with Flux models
"""

import torch
import numpy as np

# ComfyUI imports - only available when running in ComfyUI
try:
    import comfy.model_management as model_management
    import comfy.sd
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
    CATEGORY = "FluxLayerDiffuse"

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
    CATEGORY = "FluxLayerDiffuse"

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


class FluxLayerDiffuseTroubleshooter:
    """
    Diagnose and provide solutions for common Flux LayerDiffuse issues
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "issue_type": (["tensor_dimension_error", "clip_encoding_error", "model_compatibility", "general_troubleshooting"], {
                    "default": "tensor_dimension_error"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("solution",)
    FUNCTION = "diagnose_issue"
    CATEGORY = "FluxLayerDiffuse"
    
    def diagnose_issue(self, issue_type):
        """
        Provide solutions for common issues
        """
        
        if issue_type == "tensor_dimension_error":
            solution = """
=== Tensor Dimension Error Solution ===

Error: "mat1 and mat2 shapes cannot be multiplied (77x2048 and 4096x3072)"

This happens when CLIP encoding dimensions don't match Flux model expectations.

SOLUTIONS:

1. Use Flux LayerDiffuse Conditioning Fix:
   - Add "Flux LayerDiffuse Conditioning Fix" node
   - Connect your CLIP Text Encode output to it
   - Set target_length to 256 (or try 77, 512)
   - Use the fixed output in KSampler

2. Alternative Workflow:
   - Use "Flux LayerDiffuse Empty Conditioning" for negative
   - Try different CLIP models (T5 vs CLIP-L)
   - Ensure you're using Flux-compatible CLIP

3. Model Check:
   - Verify you're using Flux.1-dev model
   - Check LoRA is loaded correctly
   - Make sure model and CLIP are compatible

4. Quick Fix:
   - Try using empty string "" for negative prompt
   - Reduce batch size to 1
   - Use standard resolution (1024x1024)
"""
            
        elif issue_type == "clip_encoding_error":
            solution = """
=== CLIP Encoding Error Solution ===

Problems with CLIP text encoding for Flux models.

SOLUTIONS:

1. Use Correct CLIP Model:
   - Flux requires specific CLIP models
   - Try different CLIP loaders
   - Check model compatibility

2. Text Length Issues:
   - Keep prompts reasonable length
   - Use Conditioning Fix node
   - Try shorter prompts first

3. Encoding Format:
   - Use "Flux LayerDiffuse Conditioning Fix"
   - Adjust target_length parameter
   - Check tensor shapes in console
"""
            
        elif issue_type == "model_compatibility":
            solution = """
=== Model Compatibility Issues ===

Ensuring all models work together correctly.

CHECKLIST:

1. Flux Model:
   ✓ Use Flux.1-dev model
   ✓ Place in models/diffusion_models/
   ✓ Load with "Load Diffusion Model"

2. LoRA:
   ✓ Use layerlora.safetensors
   ✓ Place in models/loras/
   ✓ Load with strength 1.0

3. TransparentVAE:
   ✓ Use TransparentVAE.pth
   ✓ Place in models/vae/
   ✓ Load with Standalone Loader

4. CLIP:
   ✓ Use Flux-compatible CLIP
   ✓ Check encoding dimensions
   ✓ Use conditioning fix if needed
"""
            
        elif issue_type == "general_troubleshooting":
            solution = """
=== General Troubleshooting Guide ===

Step-by-step problem solving:

1. Check Console Output:
   - Look for specific error messages
   - Note tensor shapes mentioned
   - Check file loading messages

2. Verify File Locations:
   - Use "Flux LayerDiffuse Info" node
   - Check all files are found
   - Verify file sizes are correct

3. Test Basic Workflow:
   - Start with simple prompt
   - Use standard settings
   - Test without LoRA first

4. Common Fixes:
   - Restart ComfyUI
   - Clear GPU memory
   - Reduce image size
   - Use float16 instead of bfloat16

5. Get Help:
   - Share exact error message
   - Include tensor shapes from console
   - Mention your model versions
"""
        
        else:
            solution = "Unknown issue type"
        
        return (solution,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxLayerDiffuseConditioningFix": FluxLayerDiffuseConditioningFix,
    "FluxLayerDiffuseEmptyConditioning": FluxLayerDiffuseEmptyConditioning,
    "FluxLayerDiffuseTroubleshooter": FluxLayerDiffuseTroubleshooter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLayerDiffuseConditioningFix": "🔧 Flux LayerDiffuse Conditioning Fix",
    "FluxLayerDiffuseEmptyConditioning": "⭕ Flux LayerDiffuse Empty Conditioning",
    "FluxLayerDiffuseTroubleshooter": "🩺 Flux LayerDiffuse Troubleshooter",
}
