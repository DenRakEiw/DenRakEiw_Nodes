#!/usr/bin/env python3
"""
🔥 OPTIMIZED WAN VAE LATENT UPSCALER NODE - DENRAKEIW SUPERHERO EDITION 🔥
Performance-optimierte ComfyUI Node für WAN VAE Latent Upscaling:
- Automatische Model-Erkennung
- Memory-optimiert
- Batch-Processing
- Caching
- Error-Recovery
"""

import torch
import torch.nn as nn
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any

# ComfyUI Imports
try:
    import comfy.model_management as model_management
    import comfy.utils
except ImportError:
    print("⚠️ ComfyUI imports not available - running in standalone mode")

class OptimizedWanVAEUpscalerNode:
    """
    🔥 OPTIMIZED WAN VAE LATENT UPSCALER V2.0 🔥
    
    Features:
    - Auto-detects WAN VAE vs Standard VAE latents
    - Memory-optimized processing
    - Intelligent caching
    - Batch processing support
    - Error recovery
    - Performance monitoring
    """
    
    def __init__(self):
        self.advanced_model = None      # 4-channel model
        self.wan_vae_model = None       # 16-channel model
        self.device = self._get_device()
        self.model_cache = {}
        self.performance_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print(f"🔥 DENRAKEIW SUPERHERO OPTIMIZED NODE INITIALIZED")
        print(f"💪 Device: {self.device}")
    
    def _get_device(self):
        """Intelligente Device-Erkennung"""
        try:
            return model_management.get_torch_device()
        except:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "model_type": (["auto", "advanced_4ch", "wan_vae_16ch"], {
                    "default": "auto"
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 16,
                    "step": 1
                }),
                "enable_caching": ("BOOLEAN", {
                    "default": True
                }),
                "memory_optimization": (["auto", "low", "medium", "high"], {
                    "default": "auto"
                }),
            }
        }
    
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("upscaled_latent", "performance_info")
    FUNCTION = "upscale_latent_optimized"
    CATEGORY = "latent/denrakeiw"
    DESCRIPTION = "🔥 OPTIMIZED WAN VAE LATENT UPSCALER - Maximum performance with intelligent model selection!"
    
    def _detect_latent_type(self, latent_samples: torch.Tensor) -> Tuple[str, int]:
        """Intelligente Latent-Typ Erkennung"""
        batch_size, channels, height, width = latent_samples.shape
        
        if channels == 4:
            return "standard_vae", 4
        elif channels == 16:
            return "wan_vae", 16
        elif channels == 8:
            return "sdxl_vae", 8
        else:
            # Fallback: Versuche zu adaptieren
            if channels < 8:
                return "standard_vae", 4
            else:
                return "wan_vae", 16
    
    def _get_model_cache_key(self, model_type: str, channels: int) -> str:
        """Erstelle Cache-Key für Model"""
        return f"{model_type}_{channels}ch"
    
    def _load_model_cached(self, model_type: str, channels: int):
        """Lade Model mit Caching"""
        cache_key = self._get_model_cache_key(model_type, channels)
        
        if cache_key in self.model_cache:
            self.performance_stats['cache_hits'] += 1
            return self.model_cache[cache_key]
        
        self.performance_stats['cache_misses'] += 1
        
        # Lade entsprechendes Model
        if channels == 4:
            model = self._load_advanced_model()
        elif channels == 16:
            model = self._load_wan_vae_model()
        else:
            # Fallback
            model = self._load_advanced_model()
        
        # Cache Model
        self.model_cache[cache_key] = model
        return model
    
    def _load_advanced_model(self):
        """Lade Advanced 4-Channel Model"""
        if self.advanced_model is not None:
            return self.advanced_model
        
        try:
            # Import Advanced Architecture
            from advanced_trainer import AdvancedLatentUpscaler
            
            # Erstelle Model
            self.advanced_model = AdvancedLatentUpscaler(
                input_channels=4,
                output_channels=4,
                num_residual_blocks=8
            )
            
            # Lade Weights
            model_paths = [
                "models/best_model.pth",
                "custom_nodes/denrakeiw_nodes/models/best_model.pth",
                "models/final_advanced_upscaler.pth"
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    checkpoint = torch.load(path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        self.advanced_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.advanced_model.load_state_dict(checkpoint)
                    break
            
            self.advanced_model.to(self.device)
            self.advanced_model.eval()
            
            print("✅ Advanced 4-Channel Model loaded")
            return self.advanced_model
            
        except Exception as e:
            print(f"❌ Advanced model loading failed: {e}")
            return None
    
    def _load_wan_vae_model(self):
        """Lade WAN VAE 16-Channel Model"""
        if self.wan_vae_model is not None:
            return self.wan_vae_model
        
        try:
            # Import WAN VAE Architecture
            from wan_vae_trainer import AdvancedWanVAEUpscaler
            
            # Erstelle Model
            self.wan_vae_model = AdvancedWanVAEUpscaler(
                input_channels=16,
                output_channels=16,
                base_channels=64,
                num_residual_blocks=8
            )
            
            # Lade Weights
            model_paths = [
                "wan_vae_models/best_wan_vae_upscaler.pth",
                "models/best_wan_vae_upscaler.pth",
                "custom_nodes/denrakeiw_nodes/wan_vae_models/best_wan_vae_upscaler.pth"
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    checkpoint = torch.load(path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        self.wan_vae_model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.wan_vae_model.load_state_dict(checkpoint)
                    break
            
            self.wan_vae_model.to(self.device)
            self.wan_vae_model.eval()
            
            print("✅ WAN VAE 16-Channel Model loaded")
            return self.wan_vae_model
            
        except Exception as e:
            print(f"❌ WAN VAE model loading failed: {e}")
            return None
    
    def _optimize_memory(self, memory_level: str):
        """Memory Optimization"""
        if memory_level == "auto":
            # Auto-detect basierend auf verfügbarem VRAM
            try:
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    if vram_gb >= 12:
                        memory_level = "low"
                    elif vram_gb >= 8:
                        memory_level = "medium"
                    else:
                        memory_level = "high"
                else:
                    memory_level = "high"
            except:
                memory_level = "medium"
        
        if memory_level == "high":
            torch.cuda.empty_cache()
            # Weitere Memory-Optimierungen
        
        return memory_level
    
    def _process_batch(self, latent_samples: torch.Tensor, model, batch_size: int, strength: float) -> torch.Tensor:
        """Optimiertes Batch-Processing"""
        total_samples = latent_samples.shape[0]
        
        if total_samples <= batch_size:
            # Einzelner Batch
            return self._process_single_batch(latent_samples, model, strength)
        
        # Multi-Batch Processing
        results = []
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch = latent_samples[i:end_idx]
            
            result = self._process_single_batch(batch, model, strength)
            results.append(result)
            
            # Memory cleanup zwischen Batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(results, dim=0)
    
    def _process_single_batch(self, batch: torch.Tensor, model, strength: float) -> torch.Tensor:
        """Verarbeite einzelnen Batch"""
        batch = batch.to(self.device)
        
        with torch.no_grad():
            # Model inference
            upscaled = model(batch)
            
            # Apply strength
            if strength != 1.0:
                # Blend mit simple upsampling
                simple_upscale = torch.nn.functional.interpolate(
                    batch,
                    size=(upscaled.shape[2], upscaled.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                upscaled = simple_upscale + strength * (upscaled - simple_upscale)
        
        return upscaled.cpu()
    
    def _adapt_channels(self, latent: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Intelligente Channel-Adaptation"""
        current_channels = latent.shape[1]
        
        if current_channels == target_channels:
            return latent
        
        if current_channels < target_channels:
            # Pad channels
            padding = target_channels - current_channels
            pad_tensor = torch.zeros(latent.shape[0], padding, latent.shape[2], latent.shape[3])
            return torch.cat([latent, pad_tensor], dim=1)
        else:
            # Truncate channels
            return latent[:, :target_channels, :, :]
    
    def upscale_latent_optimized(self, latent, strength=1.0, model_type="auto", batch_size=1, 
                               enable_caching=True, memory_optimization="auto"):
        """
        🔥 OPTIMIZED LATENT UPSCALING V2.0 🔥
        """
        start_time = time.time()
        
        # Memory optimization
        memory_level = self._optimize_memory(memory_optimization)
        
        # Get latent samples
        latent_samples = latent["samples"]
        original_shape = latent_samples.shape
        
        # Detect latent type
        detected_type, detected_channels = self._detect_latent_type(latent_samples)
        
        print(f"🔥 OPTIMIZED UPSCALING:")
        print(f"   Input: {original_shape}")
        print(f"   Detected: {detected_type} ({detected_channels} channels)")
        print(f"   Batch size: {batch_size}")
        print(f"   Memory level: {memory_level}")
        
        # Determine model to use
        if model_type == "auto":
            if detected_channels == 4:
                use_model_type = "advanced_4ch"
                target_channels = 4
            elif detected_channels == 16:
                use_model_type = "wan_vae_16ch"
                target_channels = 16
            else:
                # Fallback
                use_model_type = "advanced_4ch"
                target_channels = 4
        else:
            use_model_type = model_type
            target_channels = 4 if "4ch" in model_type else 16
        
        # Load model (with caching)
        if enable_caching:
            model = self._load_model_cached(use_model_type, target_channels)
        else:
            if target_channels == 4:
                model = self._load_advanced_model()
            else:
                model = self._load_wan_vae_model()
        
        if model is None:
            raise RuntimeError(f"❌ Failed to load {use_model_type} model!")
        
        # Adapt channels if needed
        adapted_latent = self._adapt_channels(latent_samples, target_channels)
        
        # Process with optimized batching
        upscaled = self._process_batch(adapted_latent, model, batch_size, strength)
        
        # Restore original channel count if adapted
        if upscaled.shape[1] != original_shape[1]:
            upscaled = self._adapt_channels(upscaled, original_shape[1])
        
        # Performance stats
        end_time = time.time()
        processing_time = end_time - start_time
        
        self.performance_stats['total_processed'] += original_shape[0]
        self.performance_stats['total_time'] += processing_time
        
        # Performance info
        avg_time_per_sample = processing_time / original_shape[0]
        total_avg = self.performance_stats['total_time'] / max(1, self.performance_stats['total_processed'])
        
        performance_info = (
            f"🔥 DENRAKEIW SUPERHERO PERFORMANCE:\n"
            f"✅ Processed: {original_shape[0]} samples in {processing_time:.2f}s\n"
            f"⚡ Speed: {avg_time_per_sample:.3f}s/sample\n"
            f"📊 Total avg: {total_avg:.3f}s/sample\n"
            f"🎯 Model: {use_model_type}\n"
            f"💾 Cache hits: {self.performance_stats['cache_hits']}\n"
            f"🔄 Memory level: {memory_level}\n"
            f"🚀 Output: {upscaled.shape}"
        )
        
        print(performance_info)
        
        return ({"samples": upscaled}, performance_info)

# Node registration
NODE_CLASS_MAPPINGS = {
    "OptimizedWanVAEUpscaler": OptimizedWanVAEUpscalerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OptimizedWanVAEUpscaler": "🔥 Optimized WAN VAE Upscaler V2.0"
}
