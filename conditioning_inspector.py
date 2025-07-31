"""
Conditioning Inspector - Check tensor dimensions
"""

import torch


class ConditioningInspector:
    """
    Inspect conditioning tensor dimensions to see if Conditioning Fix is needed
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "info")
    FUNCTION = "inspect_conditioning"
    CATEGORY = "FluxLayerDiffuse"

    def inspect_conditioning(self, conditioning):
        """
        Inspect conditioning tensor and provide information
        """
        
        info_lines = ["=== Conditioning Inspection ==="]
        
        try:
            for i, cond in enumerate(conditioning):
                cond_tensor = cond[0]
                cond_dict = cond[1] if len(cond) > 1 else {}
                
                info_lines.append(f"\nConditioning {i}:")
                info_lines.append(f"  Shape: {cond_tensor.shape}")
                info_lines.append(f"  Device: {cond_tensor.device}")
                info_lines.append(f"  Dtype: {cond_tensor.dtype}")
                
                if len(cond_tensor.shape) == 3:
                    batch, seq_len, hidden_dim = cond_tensor.shape
                    info_lines.append(f"  Batch: {batch}")
                    info_lines.append(f"  Sequence Length: {seq_len}")
                    info_lines.append(f"  Hidden Dimension: {hidden_dim}")
                    
                    # Check if dimensions are Flux-compatible
                    flux_compatible = True
                    issues = []
                    
                    if seq_len not in [77, 256, 512]:
                        flux_compatible = False
                        issues.append(f"Unusual sequence length: {seq_len}")
                    
                    if hidden_dim not in [768, 1024, 2048, 4096]:
                        flux_compatible = False
                        issues.append(f"Unusual hidden dimension: {hidden_dim}")
                    
                    # Flux typically expects 256x4096 or similar
                    if seq_len == 256 and hidden_dim == 4096:
                        info_lines.append("  ✅ FLUX COMPATIBLE (256x4096)")
                    elif seq_len == 77 and hidden_dim == 2048:
                        info_lines.append("  ⚠ SD COMPATIBLE (77x2048) - May need Conditioning Fix")
                    else:
                        info_lines.append(f"  ❓ UNKNOWN FORMAT ({seq_len}x{hidden_dim})")
                    
                    if issues:
                        info_lines.append("  Issues:")
                        for issue in issues:
                            info_lines.append(f"    - {issue}")
                
                # Check dictionary contents
                if cond_dict:
                    info_lines.append(f"  Dict keys: {list(cond_dict.keys())}")
                    if 'pooled_output' in cond_dict:
                        pooled = cond_dict['pooled_output']
                        info_lines.append(f"  Pooled shape: {pooled.shape}")
            
            # Recommendation
            info_lines.append("\n=== RECOMMENDATION ===")
            
            # Check first conditioning tensor
            if conditioning and len(conditioning[0][0].shape) == 3:
                _, seq_len, hidden_dim = conditioning[0][0].shape
                
                if seq_len == 256 and hidden_dim == 4096:
                    info_lines.append("✅ Conditioning Fix NOT needed")
                    info_lines.append("   Your CLIP is already outputting Flux-compatible dimensions")
                elif seq_len == 77 and hidden_dim == 2048:
                    info_lines.append("⚠ Conditioning Fix RECOMMENDED")
                    info_lines.append("   Use 'Flux LayerDiffuse Conditioning Fix' node")
                else:
                    info_lines.append("❓ Unknown format - test both with and without Conditioning Fix")
            
            info_text = "\n".join(info_lines)
            print(info_text)
            
            return (conditioning, info_text)
            
        except Exception as e:
            error_info = f"Error inspecting conditioning: {str(e)}"
            print(error_info)
            return (conditioning, error_info)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ConditioningInspector": ConditioningInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningInspector": "🔍 Conditioning Inspector",
}
