#!/usr/bin/env python3
"""
⚡ QUICK START - ADVANCED LATENT UPSCALER TRAINING
Ein-Klick Training mit automatischem Setup
"""

import os
import sys
import subprocess
import torch

def check_requirements():
    """Check and install the requirements"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'diffusers',
        'transformers',
        'accelerate',
        'tqdm',
        'matplotlib',
        'pillow',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {missing_packages}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ All packages installed!")
    else:
        print("✅ All requirements satisfied!")

def check_gpu():
    """Check GPU availability"""
    print("\n🔧 Checking GPU...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram:.1f} GB")
        
        if vram < 6:
            print("⚠️ Warning: Low VRAM. Consider reducing batch size.")
        
        return True
    else:
        print("❌ No GPU found. Training will be slow on CPU.")
        return False

def quick_setup():
    """Quick setup for training"""
    print("🚀 QUICK START - ADVANCED LATENT UPSCALER")
    print("=" * 50)
    
    # Check requirements
    check_requirements()
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = [
        "datasets",
        "models", 
        "logs",
        "plots"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ {dir_name}/")
    
    # Recommended settings based on hardware
    print("\n⚙️ Recommended settings:")
    if has_gpu:
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 12:
            batch_size = 32
            num_workers = 8
        elif vram >= 8:
            batch_size = 16
            num_workers = 4
        else:
            batch_size = 8
            num_workers = 2
    else:
        batch_size = 4
        num_workers = 2
    
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    
    return {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'has_gpu': has_gpu
    }

def run_training(settings):
    """Starte Training"""
    print("\n🚀 Starting training...")
    
    try:
        # Import training modules
        from train_advanced_upscaler import main as train_main
        
        # Run training
        train_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all training files are in the same directory!")
        return False
    
    except Exception as e:
        print(f"❌ Training error: {e}")
        return False
    
    return True

def main():
    """Hauptfunktion"""
    print("⚡ QUICK START TRAINING")
    print("This will automatically:")
    print("1. Check and install requirements")
    print("2. Validate VAE authenticity (NO FAKES!)")
    print("3. Download datasets (DIV2K)")
    print("4. Create and validate training data")
    print("5. Train advanced upscaler")
    print()

    response = input("Continue? (y/n): ").lower().strip()
    if response != 'y':
        print("Aborted.")
        return

    # Setup
    settings = quick_setup()

    # 🔍 KRITISCHE VALIDIERUNG
    print("\n" + "="*50)
    print("🔍 RUNNING CRITICAL PRE-TRAINING VALIDATION")
    print("="*50)

    try:
        from pre_training_checker import run_comprehensive_check
        if not run_comprehensive_check():
            print("❌ VALIDIERUNG FEHLGESCHLAGEN!")
            print("🚨 TRAINING ABGEBROCHEN!")
            return
    except ImportError:
        print("⚠️ Validation system not found, proceeding with basic checks...")

    # Training
    print("\n" + "="*50)
    print("🚀 STARTING ADVANCED TRAINING")
    print("="*50)

    success = run_training(settings)
    
    if success:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("\n📁 Results:")
        print("   models/best_model.pth - Best model")
        print("   models/final_advanced_upscaler.pth - Final model")
        print("   plots/training_history.png - Training plots")
        print("   logs/ - Training logs")
        
        print("\n🔄 To use the trained model:")
        print("1. Copy best_model.pth to your ComfyUI models directory")
        print("2. Restart ComfyUI")
        print("3. Use 'Universal Latent Upscaler' node")
        
    else:
        print("\n❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
