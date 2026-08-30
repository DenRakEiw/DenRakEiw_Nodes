#!/usr/bin/env python3
"""
🔍 PRE-TRAINING CHECKER
Runs ALL critical checks before training:
- VAE authenticity
- Dataset quality  
- Latent-Validierung
- Vorher-Nachher Paare
- Visual Tests
"""

import os
import sys
import torch
from validation_system import run_complete_validation
from dataset_preparation import LatentDatasetCreator

def check_system_requirements():
    """Check the system requirements"""
    print("🔧 CHECKING SYSTEM REQUIREMENTS")
    print("=" * 40)
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram:.1f} GB")
        
        if vram < 6:
            print("⚠️ WARNUNG: Wenig VRAM! Empfohlen: 8GB+")
            print("   Reduce batch_size to 8 or less")
        
        return True
    else:
        print("❌ KEINE GPU GEFUNDEN!")
        print("🚨 Training on CPU will be VERY slow!")
        response = input("Trotzdem fortfahren? (y/n): ").lower()
        return response == 'y'

def check_disk_space():
    """Check the available disk space"""
    print("\n💾 CHECKING DISK SPACE")
    print("=" * 40)

    try:
        # Disk-space check that also works on Windows
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)

        print(f"💾 Free space: {free_space_gb:.1f} GB")

        required_space = 15  # GB for datasets + models

        if free_space_gb < required_space:
            print(f"❌ NICHT GENUG SPEICHER!")
            print(f"   Required: {required_space} GB")
            print(f"   Available: {free_space_gb:.1f} GB")
            return False

        print(f"✅ Enough free space")
        return True

    except Exception as e:
        print(f"⚠️ Disk-space check failed: {e}")
        print("✅ Fahre trotzdem fort...")
        return True

def check_internet_connection():
    """Check the internet connection for downloads"""
    print("\n🌐 CHECKING INTERNET CONNECTION")
    print("=" * 40)
    
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            print("✅ Internet-Verbindung OK")
            print("✅ Hugging Face erreichbar")
            return True
        else:
            print(f"⚠️ Hugging Face nicht erreichbar: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Internet-Verbindung FEHLER: {e}")
        print("🚨 Without internet no models or datasets can be downloaded!")
        return False

def check_existing_datasets():
    """Check the existing datasets"""
    print("\n📊 CHECKING EXISTING DATASETS")
    print("=" * 40)
    
    train_dir = "datasets/latents/train"
    val_dir = "datasets/latents/validation"
    
    train_exists = os.path.exists(train_dir)
    val_exists = os.path.exists(val_dir)
    
    if train_exists:
        train_files = len([f for f in os.listdir(train_dir) if f.endswith('.pt')])
        print(f"✅ Training Dataset: {train_files} Latents")
    else:
        print("⚠️ Training dataset not found")
        train_files = 0
    
    if val_exists:
        val_files = len([f for f in os.listdir(val_dir) if f.endswith('.pt')])
        print(f"✅ Validation Dataset: {val_files} Latents")
    else:
        print("⚠️ Validation dataset not found")
        val_files = 0
    
    if train_files < 100:
        print("⚠️ Wenige Training-Samples! Empfohlen: 500+")
        print("   Smaller datasets can lead to overfitting")
    
    if val_files < 10:
        print("⚠️ Wenige Validation-Samples! Empfohlen: 50+")
    
    return train_exists and val_exists and train_files > 0 and val_files > 0

def create_datasets_if_needed():
    """Create the datasets if they are missing"""
    print("\n📊 DATASET-ERSTELLUNG")
    print("=" * 40)
    
    if not check_existing_datasets():
        print("📥 Erstelle neue Datasets...")
        
        try:
            creator = LatentDatasetCreator()
            train_dir, val_dir = creator.create_training_dataset(target_size=1000)
            print("✅ Datasets erfolgreich erstellt!")
            return True
        except Exception as e:
            print(f"❌ Dataset-Erstellung FEHLGESCHLAGEN: {e}")
            return False
    else:
        print("✅ Datasets bereits vorhanden")
        return True

def run_comprehensive_check():
    """Run the full set of pre-training checks"""
    print("🔍 FULL PRE-TRAINING CHECK")
    print("=" * 60)
    print("This check makes sure that:")
    print("✅ a genuine Stability AI VAE is used (no fakes!)")
    print("✅ Datasets korrekt erstellt wurden")
    print("✅ the latents are valid")
    print("✅ Vorher-Nachher Paare funktionieren")
    print("✅ the system is ready for training")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 5
    
    # 1. System-Anforderungen
    if check_system_requirements():
        checks_passed += 1
        print("✅ Check 1/5: System-Anforderungen")
    else:
        print("❌ Check 1/5: System-Anforderungen FEHLGESCHLAGEN")
        return False
    
    # 2. Speicherplatz
    if check_disk_space():
        checks_passed += 1
        print("✅ Check 2/5: Speicherplatz")
    else:
        print("❌ Check 2/5: Speicherplatz FEHLGESCHLAGEN")
        return False
    
    # 3. Internet
    if check_internet_connection():
        checks_passed += 1
        print("✅ Check 3/5: Internet-Verbindung")
    else:
        print("❌ Check 3/5: Internet-Verbindung FEHLGESCHLAGEN")
        return False
    
    # 4. Datasets
    if create_datasets_if_needed():
        checks_passed += 1
        print("✅ Check 4/5: Datasets")
    else:
        print("❌ Check 4/5: Datasets FEHLGESCHLAGEN")
        return False
    
    # 5. Full validation
    print("\n🔍 FINALE VALIDIERUNG...")
    if run_complete_validation():
        checks_passed += 1
        print("✅ Check 5/5: full validation")
    else:
        print("❌ Check 5/5: full validation FAILED")
        return False
    
    # Ergebnis
    print("\n" + "=" * 60)
    print(f"🎉 ALLE CHECKS ERFOLGREICH! ({checks_passed}/{total_checks})")
    print("✅ SYSTEM READY FOR TRAINING!")
    print("=" * 60)
    
    # Training-Empfehlungen
    print("\n🚀 TRAINING-EMPFEHLUNGEN:")
    
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 12:
            print("   batch_size = 32 (hohe VRAM)")
        elif vram >= 8:
            print("   batch_size = 16 (mittlere VRAM)")
        else:
            print("   batch_size = 8 (niedrige VRAM)")
    else:
        print("   batch_size = 4 (CPU)")
    
    print("   epochs = 200 (for the best quality)")
    print("   learning_rate = 1e-4 (proven)")
    
    return True

def main():
    """Hauptfunktion"""
    print("🔍 PRE-TRAINING CHECKER")
    print("This check makes sure everything is ready for training.")
    print()
    
    if run_comprehensive_check():
        print("\n🚀 READY FOR TRAINING!")
        print("Start the training with:")
        print("   python train_advanced_upscaler.py")
        print("oder:")
        print("   python quick_start_training.py")
    else:
        print("\n❌ TRAINING NOT POSSIBLE!")
        print("Fix the problems and run the check again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
