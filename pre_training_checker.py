#!/usr/bin/env python3
"""
🔍 PRE-TRAINING CHECKER
Führt ALLE kritischen Prüfungen vor dem Training durch:
- VAE-Authentizität
- Dataset-Qualität  
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
    """Prüfe System-Anforderungen"""
    print("🔧 SYSTEM-ANFORDERUNGEN PRÜFEN")
    print("=" * 40)
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ VRAM: {vram:.1f} GB")
        
        if vram < 6:
            print("⚠️ WARNUNG: Wenig VRAM! Empfohlen: 8GB+")
            print("   Reduziere batch_size auf 8 oder weniger")
        
        return True
    else:
        print("❌ KEINE GPU GEFUNDEN!")
        print("🚨 Training auf CPU wird SEHR langsam sein!")
        response = input("Trotzdem fortfahren? (y/n): ").lower()
        return response == 'y'

def check_disk_space():
    """Prüfe verfügbaren Speicherplatz"""
    print("\n💾 SPEICHERPLATZ PRÜFEN")
    print("=" * 40)

    try:
        # Windows-kompatible Speicherplatz-Prüfung
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)

        print(f"💾 Verfügbarer Speicher: {free_space_gb:.1f} GB")

        required_space = 15  # GB für Datasets + Models

        if free_space_gb < required_space:
            print(f"❌ NICHT GENUG SPEICHER!")
            print(f"   Benötigt: {required_space} GB")
            print(f"   Verfügbar: {free_space_gb:.1f} GB")
            return False

        print(f"✅ Genug Speicher verfügbar")
        return True

    except Exception as e:
        print(f"⚠️ Speicherplatz-Prüfung fehlgeschlagen: {e}")
        print("✅ Fahre trotzdem fort...")
        return True

def check_internet_connection():
    """Prüfe Internet-Verbindung für Downloads"""
    print("\n🌐 INTERNET-VERBINDUNG PRÜFEN")
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
        print("🚨 Ohne Internet können keine Modelle/Datasets geladen werden!")
        return False

def check_existing_datasets():
    """Prüfe existierende Datasets"""
    print("\n📊 EXISTIERENDE DATASETS PRÜFEN")
    print("=" * 40)
    
    train_dir = "datasets/latents/train"
    val_dir = "datasets/latents/validation"
    
    train_exists = os.path.exists(train_dir)
    val_exists = os.path.exists(val_dir)
    
    if train_exists:
        train_files = len([f for f in os.listdir(train_dir) if f.endswith('.pt')])
        print(f"✅ Training Dataset: {train_files} Latents")
    else:
        print("⚠️ Training Dataset nicht gefunden")
        train_files = 0
    
    if val_exists:
        val_files = len([f for f in os.listdir(val_dir) if f.endswith('.pt')])
        print(f"✅ Validation Dataset: {val_files} Latents")
    else:
        print("⚠️ Validation Dataset nicht gefunden")
        val_files = 0
    
    if train_files < 100:
        print("⚠️ Wenige Training-Samples! Empfohlen: 500+")
        print("   Kleinere Datasets können zu Overfitting führen")
    
    if val_files < 10:
        print("⚠️ Wenige Validation-Samples! Empfohlen: 50+")
    
    return train_exists and val_exists and train_files > 0 and val_files > 0

def create_datasets_if_needed():
    """Erstelle Datasets falls nötig"""
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
    """Führe umfassende Pre-Training Checks durch"""
    print("🔍 UMFASSENDE PRE-TRAINING PRÜFUNG")
    print("=" * 60)
    print("Diese Prüfung stellt sicher, dass:")
    print("✅ Echte Stability AI VAE verwendet wird (keine Fakes!)")
    print("✅ Datasets korrekt erstellt wurden")
    print("✅ Latents gültig sind")
    print("✅ Vorher-Nachher Paare funktionieren")
    print("✅ System bereit für Training ist")
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
    
    # 5. Vollständige Validierung
    print("\n🔍 FINALE VALIDIERUNG...")
    if run_complete_validation():
        checks_passed += 1
        print("✅ Check 5/5: Vollständige Validierung")
    else:
        print("❌ Check 5/5: Vollständige Validierung FEHLGESCHLAGEN")
        return False
    
    # Ergebnis
    print("\n" + "=" * 60)
    print(f"🎉 ALLE CHECKS ERFOLGREICH! ({checks_passed}/{total_checks})")
    print("✅ SYSTEM BEREIT FÜR TRAINING!")
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
    
    print("   epochs = 200 (für beste Qualität)")
    print("   learning_rate = 1e-4 (bewährt)")
    
    return True

def main():
    """Hauptfunktion"""
    print("🔍 PRE-TRAINING CHECKER")
    print("Dieser Check stellt sicher, dass alles für das Training bereit ist.")
    print()
    
    if run_comprehensive_check():
        print("\n🚀 BEREIT FÜR TRAINING!")
        print("Starte das Training mit:")
        print("   python train_advanced_upscaler.py")
        print("oder:")
        print("   python quick_start_training.py")
    else:
        print("\n❌ TRAINING NICHT MÖGLICH!")
        print("Behebe die Probleme und führe den Check erneut aus.")
        sys.exit(1)

if __name__ == "__main__":
    main()
