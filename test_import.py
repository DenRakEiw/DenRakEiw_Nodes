#!/usr/bin/env python3
"""
Test script to verify Flux LayerDiffuse nodes can be imported correctly
"""

import sys
import traceback

def test_imports():
    """Test if all required modules can be imported"""
    
    print("🧪 Testing Flux LayerDiffuse Node Imports...")
    print("=" * 50)
    
    # Test basic imports
    try:
        import torch
        print("✅ PyTorch:", torch.__version__)
    except ImportError as e:
        print("❌ PyTorch not available:", e)
        return False
    
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ NumPy not available:", e)
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow: Available")
    except ImportError as e:
        print("❌ Pillow not available:", e)
        return False
    
    # Test diffusers
    try:
        import diffusers
        print("✅ Diffusers:", diffusers.__version__)
    except ImportError as e:
        print("❌ Diffusers not available:", e)
        print("   Install with: pip install diffusers==0.32.2")
        return False
    
    # Test transformers
    try:
        import transformers
        print("✅ Transformers:", transformers.__version__)
    except ImportError as e:
        print("❌ Transformers not available:", e)
        print("   Install with: pip install transformers>=4.48.0")
        return False
    
    # Test safetensors
    try:
        import safetensors
        print("✅ Safetensors: Available")
    except ImportError as e:
        print("❌ Safetensors not available:", e)
        print("   Install with: pip install safetensors")
        return False
    
    # Test our custom modules
    print("\n🔧 Testing Custom Modules...")
    print("-" * 30)
    
    try:
        from transparent_vae import TransparentVAE, UNet1024, LatentTransparencyOffsetEncoder
        print("✅ TransparentVAE module: OK")
    except ImportError as e:
        print("❌ TransparentVAE module failed:", e)
        traceback.print_exc()
        return False
    
    try:
        from flux_layerdiffuse_loader import FluxLayerDiffuseLoader, FluxLayerDiffuseModelManager
        print("✅ Loader module: OK")
    except ImportError as e:
        print("❌ Loader module failed:", e)
        traceback.print_exc()
        return False
    
    try:
        from flux_layerdiffuse_t2i import FluxLayerDiffuseT2I, FluxLayerDiffuseT2IAdvanced
        print("✅ T2I module: OK")
    except ImportError as e:
        print("❌ T2I module failed:", e)
        traceback.print_exc()
        return False
    
    try:
        from flux_layerdiffuse_i2i import FluxLayerDiffuseI2I, FluxLayerDiffuseI2IAdvanced
        print("✅ I2I module: OK")
    except ImportError as e:
        print("❌ I2I module failed:", e)
        traceback.print_exc()
        return False
    
    # Test model files
    print("\n📁 Checking Model Files...")
    print("-" * 25)
    
    import os
    
    transparent_vae_path = "models/TransparentVAE.pth"
    lora_path = "models/layerlora.safetensors"
    
    if os.path.exists(transparent_vae_path):
        size_mb = os.path.getsize(transparent_vae_path) / (1024 * 1024)
        print(f"✅ TransparentVAE.pth: {size_mb:.1f} MB")
    else:
        print("❌ TransparentVAE.pth: Not found")
        print("   Download from: https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse")
    
    if os.path.exists(lora_path):
        size_mb = os.path.getsize(lora_path) / (1024 * 1024)
        print(f"✅ layerlora.safetensors: {size_mb:.1f} MB")
    else:
        print("❌ layerlora.safetensors: Not found")
        print("   Download from: https://huggingface.co/RedAIGC/Flux-version-LayerDiffuse")
    
    print("\n🎉 All tests passed! Flux LayerDiffuse nodes should work correctly.")
    return True

def test_node_registration():
    """Test if nodes can be registered properly"""

    print("\n🔗 Testing Node Registration...")
    print("-" * 30)

    try:
        # Test individual node classes
        node_classes = [
            ("FluxLayerDiffuseLoader", "flux_layerdiffuse_loader"),
            ("FluxLayerDiffuseModelManager", "flux_layerdiffuse_loader"),
            ("FluxLayerDiffuseT2I", "flux_layerdiffuse_t2i"),
            ("FluxLayerDiffuseT2IAdvanced", "flux_layerdiffuse_t2i"),
            ("FluxLayerDiffuseI2I", "flux_layerdiffuse_i2i"),
            ("FluxLayerDiffuseI2IAdvanced", "flux_layerdiffuse_i2i"),
        ]

        loaded_classes = {}

        for class_name, module_name in node_classes:
            try:
                module = __import__(module_name)
                if hasattr(module, class_name):
                    node_class = getattr(module, class_name)
                    loaded_classes[class_name] = node_class
                    print(f"✅ {class_name}: Available")
                else:
                    print(f"❌ {class_name}: Not found in {module_name}")
            except ImportError as e:
                print(f"❌ {class_name}: Import failed - {e}")

        print(f"\n✅ Successfully loaded {len(loaded_classes)} node classes")

        # Test if classes have required methods
        for class_name, node_class in loaded_classes.items():
            if hasattr(node_class, 'INPUT_TYPES'):
                print(f"   - {class_name}: Has INPUT_TYPES ✓")
            else:
                print(f"   - {class_name}: Missing INPUT_TYPES ❌")

        return len(loaded_classes) > 0

    except Exception as e:
        print(f"❌ Node registration test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Flux LayerDiffuse Node Test Suite")
    print("=" * 50)
    
    success = True
    
    # Run import tests
    if not test_imports():
        success = False
    
    # Run registration tests
    if not test_node_registration():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Your Flux LayerDiffuse nodes are ready to use!")
        print("\nNext steps:")
        print("1. Restart ComfyUI")
        print("2. Look for 'FluxLayerDiffuse' category in the node menu")
        print("3. Start with the 'Flux LayerDiffuse Model Manager' node")
    else:
        print("❌ SOME TESTS FAILED! Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Download required model files")
        print("3. Check Python environment compatibility")
    
    print("=" * 50)
