import torch
import torch.nn.functional as F
import comfy.model_management
import comfy.utils
import numpy as np
import math

# Try to import kornia for advanced color operations
try:
    import kornia
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Kornia not available for LatentImageAdjust, some features will be limited")

class LatentImageAdjust:
    """
    Latent Image Adjust Node
    Applies image adjustments (hue, saturation, brightness, contrast, sharpness) 
    directly in the latent space for better performance and integration
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "hue": ("FLOAT", { "default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, }),
                "saturation": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, }),
                "brightness": ("FLOAT", { "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, }),
                "contrast": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, }),
                "sharpness": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05, }),
                "device": (["auto", "cpu", "gpu"],),
                "batch_size": ("INT", { "default": 0, "min": 0, "max": 1024, "step": 1, }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "latent/adjust"

    def execute(self, latent, hue, saturation, brightness, contrast, sharpness, device, batch_size):
        print(f"=== LatentImageAdjust Debug ===")
        print(f"Hue: {hue}°, Saturation: {saturation}, Brightness: {brightness}")
        print(f"Contrast: {contrast}, Sharpness: {sharpness}")
        
        # Device management
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        # Extract latent tensors and handle shape
        latent_samples = latent["samples"]
        original_shape = latent_samples.shape
        
        print(f"Original shape: {original_shape}")
        
        # Handle 5D tensors by squeezing
        working_latent = latent_samples
        while len(working_latent.shape) > 4:
            for dim in range(len(working_latent.shape) - 1, 0, -1):
                if working_latent.shape[dim] == 1:
                    working_latent = working_latent.squeeze(dim)
                    break
            else:
                working_latent = working_latent.squeeze()
                break
        
        print(f"Working shape: {working_latent.shape}")
        print(f"Latent range: {working_latent.min().item():.3f} to {working_latent.max().item():.3f}")
        
        # Move to device
        working_latent = working_latent.to(device)
        
        # Handle batch processing
        if batch_size == 0 or batch_size > working_latent.shape[0]:
            batch_size = working_latent.shape[0]

        # Process latent in batches
        latent_batch = torch.split(working_latent, batch_size, dim=0)
        output = []

        for batch in latent_batch:
            batch = batch.to(device)
            
            # Apply adjustments in sequence
            adjusted = batch.clone()
            
            # 1. Brightness (additive)
            if brightness != 0.0:
                adjusted = self.adjust_brightness_latent(adjusted, brightness)
            
            # 2. Contrast (multiplicative around mean)
            if contrast != 1.0:
                adjusted = self.adjust_contrast_latent(adjusted, contrast)
            
            # 3. Hue and Saturation (requires color space conversion)
            if hue != 0.0 or saturation != 1.0:
                adjusted = self.adjust_hue_saturation_latent(adjusted, hue, saturation)
            
            # 4. Sharpness (spatial filtering)
            if sharpness != 1.0:
                adjusted = self.adjust_sharpness_latent(adjusted, sharpness)
            
            output.append(adjusted.to(comfy.model_management.intermediate_device()))

        # Combine results
        adjusted_samples = torch.cat(output, dim=0)
        
        # Restore original shape if needed
        if adjusted_samples.shape != original_shape:
            print(f"Restoring shape from {adjusted_samples.shape} to {original_shape}")
            if len(original_shape) == 5 and len(adjusted_samples.shape) == 4:
                for dim in range(1, len(original_shape)):
                    if original_shape[dim] == 1:
                        adjusted_samples = adjusted_samples.unsqueeze(dim)
                        break
        
        print(f"Final range: {adjusted_samples.min().item():.3f} to {adjusted_samples.max().item():.3f}")
        print(f"Final shape: {adjusted_samples.shape}")
        
        # Create output latent dict
        result = latent.copy()
        result["samples"] = adjusted_samples
        
        return (result,)

    def adjust_brightness_latent(self, latent, brightness):
        """Adjust brightness in latent space"""
        print(f"Applying brightness: {brightness}")
        
        # Scale brightness for latent space
        latent_brightness = brightness * 0.5  # Reduce intensity for latents
        
        # Simple additive brightness
        adjusted = latent + latent_brightness
        
        return adjusted

    def adjust_contrast_latent(self, latent, contrast):
        """Adjust contrast in latent space"""
        print(f"Applying contrast: {contrast}")
        
        # Calculate mean for each channel
        mean = latent.mean(dim=[2, 3], keepdim=True)
        
        # Apply contrast around the mean
        adjusted = (latent - mean) * contrast + mean
        
        return adjusted

    def adjust_hue_saturation_latent(self, latent, hue, saturation):
        """Adjust hue and saturation in latent space"""
        print(f"Applying hue: {hue}°, saturation: {saturation}")
        
        # For latents, we'll work with the first 3 channels as "RGB-like"
        if latent.shape[1] >= 3:
            rgb_channels = latent[:, :3, :, :]
            extra_channels = latent[:, 3:, :, :] if latent.shape[1] > 3 else None
            
            if KORNIA_AVAILABLE and hue != 0.0:
                # Use kornia for hue adjustment if available
                adjusted_rgb = self.adjust_hue_kornia(rgb_channels, hue)
            else:
                # Fallback: simple channel rotation for hue
                adjusted_rgb = self.adjust_hue_simple(rgb_channels, hue)
            
            # Apply saturation
            if saturation != 1.0:
                adjusted_rgb = self.adjust_saturation_latent(adjusted_rgb, saturation)
            
            # Recombine channels
            if extra_channels is not None:
                adjusted = torch.cat([adjusted_rgb, extra_channels], dim=1)
            else:
                adjusted = adjusted_rgb[:, :latent.shape[1], :, :]
        else:
            # For non-RGB latents, apply simple saturation-like scaling
            adjusted = latent * saturation
        
        return adjusted

    def adjust_hue_kornia(self, rgb_latent, hue_degrees):
        """Adjust hue using kornia (if available)"""
        try:
            # Normalize to 0-1 for kornia
            latent_min = rgb_latent.min()
            latent_max = rgb_latent.max()
            latent_range = latent_max - latent_min + 1e-8
            normalized = (rgb_latent - latent_min) / latent_range
            
            # Convert to HSV, adjust hue, convert back
            hsv = kornia.color.rgb_to_hsv(normalized)
            hsv[:, 0, :, :] = (hsv[:, 0, :, :] + hue_degrees / 360.0) % 1.0
            adjusted_norm = kornia.color.hsv_to_rgb(hsv)
            
            # Denormalize
            adjusted = adjusted_norm * latent_range + latent_min
            return adjusted
        except Exception as e:
            print(f"Kornia hue adjustment failed: {e}, using simple method")
            return self.adjust_hue_simple(rgb_latent, hue_degrees)

    def adjust_hue_simple(self, rgb_latent, hue_degrees):
        """Simple hue adjustment by channel rotation"""
        if hue_degrees == 0.0:
            return rgb_latent
        
        # Convert degrees to radians
        hue_rad = math.radians(hue_degrees)
        
        # Simple rotation matrix for RGB channels
        cos_h = math.cos(hue_rad)
        sin_h = math.sin(hue_rad)
        
        # Apply rotation (simplified)
        r, g, b = rgb_latent[:, 0:1, :, :], rgb_latent[:, 1:2, :, :], rgb_latent[:, 2:3, :, :]
        
        # Simple hue shift
        factor = abs(sin_h) * 0.3  # Reduce intensity
        r_new = r + factor * (g - b)
        g_new = g + factor * (b - r)
        b_new = b + factor * (r - g)
        
        adjusted = torch.cat([r_new, g_new, b_new], dim=1)
        return adjusted

    def adjust_saturation_latent(self, rgb_latent, saturation):
        """Adjust saturation in latent space"""
        # Calculate grayscale (luminance) for each pixel
        # Use standard RGB to grayscale weights adapted for latents
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_latent.device).view(1, 3, 1, 1)
        gray = (rgb_latent * weights).sum(dim=1, keepdim=True)
        
        # Blend between grayscale and original based on saturation
        adjusted = gray + saturation * (rgb_latent - gray)
        
        return adjusted

    def adjust_sharpness_latent(self, latent, sharpness):
        """Adjust sharpness in latent space using unsharp masking"""
        print(f"Applying sharpness: {sharpness}")
        
        if sharpness == 1.0:
            return latent
        
        # Create blur kernel for unsharp masking
        kernel_size = 3
        sigma = 0.8
        
        # Gaussian blur kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma, latent.device)
        
        # Apply blur to each channel
        blurred = torch.zeros_like(latent)
        for c in range(latent.shape[1]):
            channel = latent[:, c:c+1, :, :]
            padded = F.pad(channel, (1, 1, 1, 1), mode='reflect')
            blurred[:, c:c+1, :, :] = F.conv2d(padded, kernel, padding=0)
        
        # Unsharp masking: original + (original - blurred) * strength
        if sharpness > 1.0:
            # Sharpen
            strength = (sharpness - 1.0) * 0.5  # Reduce intensity for latents
            adjusted = latent + strength * (latent - blurred)
        else:
            # Blur (sharpness < 1.0)
            blend_factor = sharpness
            adjusted = blend_factor * latent + (1 - blend_factor) * blurred
        
        return adjusted

    def create_gaussian_kernel(self, kernel_size, sigma, device):
        """Create a Gaussian blur kernel"""
        coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
        coords -= kernel_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        
        # Create 2D kernel
        kernel_2d = g[:, None] * g[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Reshape for conv2d
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        return kernel
