"""
Transparent Image Nodes for ComfyUI
Handles saving and previewing transparent RGBA images
"""

import torch
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

# ComfyUI imports - only available when running in ComfyUI
try:
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False


class SaveTransparentImage:
    """
    Save transparent RGBA images as PNG files
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "transparent_",
                    "multiline": False
                }),
            },
            "optional": {
                "save_metadata": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_transparent_images"
    CATEGORY = "FluxLayerDiffuse"
    OUTPUT_NODE = True

    def save_transparent_images(self, images, filename_prefix, save_metadata=True):
        """
        Save transparent images as PNG files
        """
        
        try:
            # Get output directory
            if COMFY_AVAILABLE:
                output_dir = folder_paths.get_output_directory()
            else:
                output_dir = "output"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Handle different input formats
            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            elif isinstance(images, np.ndarray):
                images_np = images
            else:
                raise ValueError(f"Unsupported image type: {type(images)}")
            
            print(f"Saving transparent images: {images_np.shape}")
            
            saved_paths = []
            
            # Process each image in the batch
            for i, image_np in enumerate(images_np):
                # Ensure image is in correct format [H, W, C]
                if len(image_np.shape) == 4:
                    image_np = image_np[0]  # Remove batch dimension if present
                
                # Convert to 0-255 range if needed
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
                
                # Handle different channel counts
                if image_np.shape[-1] == 4:
                    # RGBA - perfect for transparent PNG
                    mode = "RGBA"
                elif image_np.shape[-1] == 3:
                    # RGB - add alpha channel
                    alpha = np.ones((image_np.shape[0], image_np.shape[1], 1), dtype=np.uint8) * 255
                    image_np = np.concatenate([image_np, alpha], axis=2)
                    mode = "RGBA"
                else:
                    raise ValueError(f"Unsupported channel count: {image_np.shape[-1]}")
                
                # Create PIL Image
                pil_image = Image.fromarray(image_np, mode=mode)

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if len(images_np) > 1:
                    filename = f"{filename_prefix}{timestamp}_{i:03d}.png"
                else:
                    filename = f"{filename_prefix}{timestamp}.png"

                filepath = os.path.join(output_dir, filename)

                # Prepare PNG info for metadata embedding
                pnginfo = None
                if save_metadata:
                    from PIL.PngImagePlugin import PngInfo
                    pnginfo = PngInfo()

                    # Add metadata to PNG
                    pnginfo.add_text("Software", "ComfyUI Flux LayerDiffuse")
                    pnginfo.add_text("Generator", "Flux LayerDiffuse")
                    pnginfo.add_text("Creation Time", timestamp)
                    pnginfo.add_text("Format", "PNG")
                    pnginfo.add_text("Mode", mode)
                    pnginfo.add_text("Size", f"{pil_image.size[0]}x{pil_image.size[1]}")
                    pnginfo.add_text("Channels", str(image_np.shape[-1]))
                    pnginfo.add_text("Has Transparency", "true")
                    pnginfo.add_text("Workflow", "Flux LayerDiffuse Transparent Generation")

                    # Add technical details
                    pnginfo.add_text("Color Space", "sRGB")
                    pnginfo.add_text("Bit Depth", "8")
                    pnginfo.add_text("Compression", "PNG")

                # Save as PNG with embedded metadata
                pil_image.save(filepath, "PNG", optimize=True, pnginfo=pnginfo)

                saved_paths.append(filepath)
                print(f"✓ Saved transparent image: {filepath}")
                if save_metadata:
                    print(f"  ✓ Embedded metadata in PNG")
            
            # For ComfyUI display, we need to return the images in the correct format
            # Convert back to ComfyUI format for display
            display_images = []

            for i, image_np in enumerate(images_np):
                # Ensure image is in correct format [H, W, C]
                if len(image_np.shape) == 4:
                    image_np = image_np[0]  # Remove batch dimension if present

                # Convert to 0-1 range for ComfyUI display
                if image_np.max() > 1.0:
                    image_np = image_np.astype(np.float32) / 255.0
                else:
                    image_np = image_np.astype(np.float32)

                # Handle different channel counts for display
                if image_np.shape[-1] == 4:
                    # RGBA - composite over white background for display
                    rgb = image_np[:, :, :3]
                    alpha = image_np[:, :, 3:4]
                    # Composite over white background
                    display_image = rgb * alpha + (1 - alpha)
                elif image_np.shape[-1] == 3:
                    # RGB
                    display_image = image_np
                else:
                    raise ValueError(f"Unsupported channel count: {image_np.shape[-1]}")

                display_images.append(display_image)

            # Convert to tensor for ComfyUI
            if display_images:
                display_tensor = torch.from_numpy(np.stack(display_images))
                # Store for ComfyUI preview system
                self.display_images = display_tensor

            print(f"✓ Saved and prepared {len(saved_paths)} images for display")

            return {"ui": {"images": [{"filename": os.path.basename(path), "subfolder": "", "type": "output"} for path in saved_paths]}}
            
        except Exception as e:
            print(f"Error saving transparent images: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to save transparent images: {str(e)}")


class PreviewTransparentImage:
    """
    Preview transparent RGBA images in ComfyUI
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "background_color": (["transparent", "white", "black", "checkerboard"], {
                    "default": "checkerboard"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preview_transparent_images"
    CATEGORY = "FluxLayerDiffuse"
    OUTPUT_NODE = True

    def preview_transparent_images(self, images, background_color):
        """
        Preview transparent images with different backgrounds
        """
        
        try:
            # Handle different input formats
            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            elif isinstance(images, np.ndarray):
                images_np = images
            else:
                raise ValueError(f"Unsupported image type: {type(images)}")
            
            print(f"Previewing transparent images: {images_np.shape}")

            # Get temporary directory for preview
            if COMFY_AVAILABLE:
                temp_dir = folder_paths.get_temp_directory()
            else:
                temp_dir = "temp"

            os.makedirs(temp_dir, exist_ok=True)

            preview_paths = []

            # Process each image
            for i, image_np in enumerate(images_np):
                # Ensure image is in correct format [H, W, C]
                if len(image_np.shape) == 4:
                    image_np = image_np[0]  # Remove batch dimension if present

                # Convert to 0-255 range if needed
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

                # Handle different channel counts
                if image_np.shape[-1] == 4:
                    # RGBA
                    rgba_image = image_np
                elif image_np.shape[-1] == 3:
                    # RGB - add alpha channel
                    alpha = np.ones((image_np.shape[0], image_np.shape[1], 1), dtype=np.uint8) * 255
                    rgba_image = np.concatenate([image_np, alpha], axis=2)
                else:
                    raise ValueError(f"Unsupported channel count: {image_np.shape[-1]}")
                
                # Create background based on selection
                h, w = rgba_image.shape[:2]
                
                if background_color == "white":
                    background = np.ones((h, w, 3), dtype=np.uint8) * 255
                elif background_color == "black":
                    background = np.zeros((h, w, 3), dtype=np.uint8)
                elif background_color == "checkerboard":
                    # Create checkerboard pattern
                    checker_size = 16
                    background = np.zeros((h, w, 3), dtype=np.uint8)
                    for y in range(0, h, checker_size):
                        for x in range(0, w, checker_size):
                            if (x // checker_size + y // checker_size) % 2:
                                background[y:y+checker_size, x:x+checker_size] = 200
                            else:
                                background[y:y+checker_size, x:x+checker_size] = 255
                elif background_color == "transparent":
                    # Keep as RGBA for transparent preview
                    pil_image = Image.fromarray(rgba_image, mode="RGBA")
                    
                    # Save preview
                    preview_filename = f"transparent_preview_{i:03d}.png"
                    preview_path = os.path.join(temp_dir, preview_filename)
                    pil_image.save(preview_path, "PNG")
                    preview_paths.append(preview_path)
                    continue
                
                # Composite RGBA over background
                rgb = rgba_image[:, :, :3]
                alpha = rgba_image[:, :, 3:4] / 255.0
                
                # Alpha blending
                composited = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
                
                # Create PIL image
                pil_image = Image.fromarray(composited, mode="RGB")
                
                # Save preview
                preview_filename = f"transparent_preview_{background_color}_{i:03d}.png"
                preview_path = os.path.join(temp_dir, preview_filename)
                pil_image.save(preview_path, "PNG")
                preview_paths.append(preview_path)
            
            print(f"✓ Created {len(preview_paths)} preview images")

            # For ComfyUI display, prepare the composited images
            display_images = []

            for i, image_np in enumerate(images_np):
                # Ensure image is in correct format [H, W, C]
                if len(image_np.shape) == 4:
                    image_np = image_np[0]  # Remove batch dimension if present

                # Convert to 0-1 range for ComfyUI display
                if image_np.max() > 1.0:
                    image_np = image_np.astype(np.float32) / 255.0
                else:
                    image_np = image_np.astype(np.float32)

                # Handle different channel counts for display
                if image_np.shape[-1] == 4:
                    # RGBA - composite based on background_color
                    rgb = image_np[:, :, :3]
                    alpha = image_np[:, :, 3:4]

                    if background_color == "white":
                        background = np.ones_like(rgb)
                    elif background_color == "black":
                        background = np.zeros_like(rgb)
                    elif background_color == "checkerboard":
                        # Create checkerboard pattern
                        h, w = rgb.shape[:2]
                        checker_size = 16
                        background = np.zeros_like(rgb)
                        for y in range(0, h, checker_size):
                            for x in range(0, w, checker_size):
                                if (x // checker_size + y // checker_size) % 2:
                                    background[y:y+checker_size, x:x+checker_size] = 0.8
                                else:
                                    background[y:y+checker_size, x:x+checker_size] = 1.0
                    else:  # transparent
                        background = np.ones_like(rgb)  # fallback to white

                    # Composite
                    display_image = rgb * alpha + background * (1 - alpha)
                elif image_np.shape[-1] == 3:
                    # RGB
                    display_image = image_np
                else:
                    raise ValueError(f"Unsupported channel count: {image_np.shape[-1]}")

                display_images.append(display_image)

            # Convert to tensor for ComfyUI
            if display_images:
                display_tensor = torch.from_numpy(np.stack(display_images))
                # Store for ComfyUI preview system
                self.display_images = display_tensor

            return {"ui": {"images": [{"filename": os.path.basename(path), "subfolder": "", "type": "temp"} for path in preview_paths]}}

        except Exception as e:
            print(f"Error previewing transparent images: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty dict on error
            return {"ui": {"images": []}}


class TransparentImageInfo:
    """
    Display information about transparent images
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "analyze_transparent_images"
    CATEGORY = "FluxLayerDiffuse"

    def analyze_transparent_images(self, images):
        """
        Analyze transparent images and provide information
        """
        
        try:
            # Handle different input formats
            if isinstance(images, torch.Tensor):
                images_np = images.cpu().numpy()
            elif isinstance(images, np.ndarray):
                images_np = images
            else:
                return (f"Unsupported image type: {type(images)}",)
            
            info_lines = ["=== Transparent Image Analysis ==="]
            
            info_lines.append(f"Batch size: {len(images_np)}")
            info_lines.append(f"Input shape: {images_np.shape}")
            info_lines.append(f"Input dtype: {images_np.dtype}")
            
            # Analyze each image
            for i, image_np in enumerate(images_np):
                if len(image_np.shape) == 4:
                    image_np = image_np[0]  # Remove batch dimension
                
                h, w, c = image_np.shape
                info_lines.append(f"\nImage {i}:")
                info_lines.append(f"  Size: {w}x{h}")
                info_lines.append(f"  Channels: {c}")
                
                # Check value range
                min_val, max_val = image_np.min(), image_np.max()
                info_lines.append(f"  Value range: {min_val:.3f} - {max_val:.3f}")
                
                if c == 4:
                    # Analyze alpha channel
                    alpha = image_np[:, :, 3]
                    alpha_min, alpha_max = alpha.min(), alpha.max()
                    alpha_mean = alpha.mean()
                    
                    info_lines.append(f"  Alpha range: {alpha_min:.3f} - {alpha_max:.3f}")
                    info_lines.append(f"  Alpha mean: {alpha_mean:.3f}")
                    
                    # Check transparency
                    if alpha_min < 0.1:
                        info_lines.append("  ✅ Has transparent pixels")
                    else:
                        info_lines.append("  ⚠ No transparent pixels found")
                    
                    if alpha_max > 0.9:
                        info_lines.append("  ✅ Has opaque pixels")
                    else:
                        info_lines.append("  ⚠ No opaque pixels found")
                
                elif c == 3:
                    info_lines.append("  ⚠ RGB only - no alpha channel")
                else:
                    info_lines.append(f"  ❓ Unusual channel count: {c}")
            
            info_text = "\n".join(info_lines)
            print(info_text)
            
            return (info_text,)
            
        except Exception as e:
            error_info = f"Error analyzing transparent images: {str(e)}"
            print(error_info)
            return (error_info,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "SaveTransparentImage": SaveTransparentImage,
    "PreviewTransparentImage": PreviewTransparentImage,
    "TransparentImageInfo": TransparentImageInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveTransparentImage": "💾 Save Transparent Image",
    "PreviewTransparentImage": "👁️ Preview Transparent Image",
    "TransparentImageInfo": "📊 Transparent Image Info",
}
