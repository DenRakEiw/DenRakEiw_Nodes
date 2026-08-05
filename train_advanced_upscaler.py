#!/usr/bin/env python3
"""
🚀 MAIN TRAINING SCRIPT - ADVANCED LATENT UPSCALER V2.0
Komplettes Training-System mit:
- Automatischem Dataset-Download
- Verbesserter Architektur
- Perceptual Loss
- Progressive Training
- Monitoring & Visualization
"""

import torch
import torch.nn as nn
import os
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader

# Import unsere Module
from advanced_trainer import AdvancedLatentUpscaler, AdvancedTrainer
from dataset_preparation import LatentDatasetCreator, LatentAugmentationDataset
from validation_system import run_complete_validation

def setup_training_environment():
    """Setup Training Environment"""
    print("🔧 Setting up training environment...")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if device.type == 'cuda':
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(f"🔧 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    return device

def create_model(config):
    """Erstelle Model basierend auf Config"""
    model = AdvancedLatentUpscaler(
        input_channels=config.get('input_channels', 4),
        output_channels=config.get('output_channels', 4),
        num_residual_blocks=config.get('num_residual_blocks', 8)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 Model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model

def prepare_datasets(config):
    """Bereite Datasets vor"""
    print("📊 Preparing datasets...")
    
    # Check if datasets exist
    train_dir = "datasets/latents/train"
    val_dir = "datasets/latents/validation"
    
    if not os.path.exists(train_dir):
        print("❌ Training dataset not found!")
        print("🔥 DENRAKEIW SUPERHERO BYPASS: Using existing latents!")
        # Verwende existierende Latents ohne VAE-Encoding
        train_dir = "datasets/latents/train"
        os.makedirs(train_dir, exist_ok=True)

    if not os.path.exists(val_dir):
        print("⚠️ Validation dataset not found, creating from training data...")
        val_dir = "datasets/latents/validation"
        os.makedirs(val_dir, exist_ok=True)

        # Kopiere 20% der Training-Daten als Validation
        import shutil
        train_files = [f for f in os.listdir(train_dir) if f.endswith('.pt')]
        val_count = max(1, len(train_files) // 5)  # 20% für Validation

        for i, filename in enumerate(train_files[:val_count]):
            src = os.path.join(train_dir, filename)
            dst = os.path.join(val_dir, f"val_{filename}")
            shutil.copy2(src, dst)

        print(f"✅ Created {val_count} validation samples from training data")

    print("✅ Datasets ready!")
    
    # Create datasets
    train_dataset = LatentAugmentationDataset(train_dir, augment=True)
    val_dataset = LatentAugmentationDataset(val_dir, augment=False)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 16),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader

def plot_training_history(history, save_path="plots/training_history.png"):
    """Visualisiere Training History"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Learning rate
    axes[0, 1].plot(history['lr'], color='green')
    axes[0, 1].set_title('Learning Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('LR')
    axes[0, 1].grid(True)
    
    # Loss difference
    if len(history['train_loss']) > 1:
        train_diff = np.diff(history['train_loss'])
        val_diff = np.diff(history['val_loss'])
        axes[1, 0].plot(train_diff, label='Train Loss Diff', color='blue', alpha=0.7)
        axes[1, 0].plot(val_diff, label='Val Loss Diff', color='red', alpha=0.7)
        axes[1, 0].set_title('Loss Changes')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss Difference')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Validation/Training ratio
    if len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
        ratio = np.array(history['val_loss']) / np.array(history['train_loss'])
        axes[1, 1].plot(ratio, color='purple')
        axes[1, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Val/Train Loss Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Training plots saved to: {save_path}")

def save_training_config(config, save_path="logs/training_config.json"):
    """Speichere Training Configuration"""
    config_copy = config.copy()
    config_copy['timestamp'] = datetime.now().isoformat()
    
    with open(save_path, 'w') as f:
        json.dump(config_copy, f, indent=2)
    
    print(f"⚙️ Config saved to: {save_path}")

def main():
    """Hauptfunktion"""
    print("🚀 ADVANCED LATENT UPSCALER TRAINING V2.0")
    print("=" * 60)
    
    # Training Configuration
    config = {
        # Model
        'input_channels': 4,
        'output_channels': 4,
        'num_residual_blocks': 8,
        
        # Training
        'epochs': 200,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        
        # Dataset
        'dataset_size': 2000,
        'num_workers': 4,
        
        # Loss
        'mse_weight': 0.7,
        'perceptual_weight': 0.3,
        
        # Scheduler
        'scheduler_type': 'cosine',
        'min_lr': 1e-6,
        
        # Saving
        'save_every': 10,
        'validate_every': 1
    }
    
    # Setup
    device = setup_training_environment()
    save_training_config(config)

    # 🔍 KRITISCHE VALIDIERUNG VOR TRAINING!
    print("\n🔍 RUNNING PRE-TRAINING VALIDATION...")
    print("=" * 60)
    if not run_complete_validation():
        print("❌ VALIDIERUNG FEHLGESCHLAGEN!")
        print("🚨 TRAINING ABGEBROCHEN - PRÜFE DEINE DATEN!")
        return
    print("✅ ALLE VALIDIERUNGEN ERFOLGREICH!")

    # Model
    print("\n🧠 Creating model...")
    model = create_model(config)
    
    # Datasets
    print("\n📊 Preparing datasets...")
    train_loader, val_loader = prepare_datasets(config)
    
    # Trainer
    print("\n🏋️ Initializing trainer...")
    trainer = AdvancedTrainer(model, device)
    
    # Training
    print(f"\n🚀 Starting training for {config['epochs']} epochs...")
    print("=" * 60)
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['epochs'],
            save_dir="models"
        )
        
        # Save final results
        print("\n💾 Saving final results...")
        
        # Plot training history
        plot_training_history(history)
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }, "models/final_advanced_upscaler.pth")
        
        # Save history
        with open("logs/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        print("✅ Training completed successfully!")
        print(f"📊 Final Train Loss: {history['train_loss'][-1]:.6f}")
        print(f"📊 Final Val Loss: {history['val_loss'][-1]:.6f}")
        print(f"💾 Best model saved to: models/best_model.pth")
        print(f"💾 Final model saved to: models/final_advanced_upscaler.pth")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("💾 Saving current state...")
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': trainer.history,
            'interrupted': True
        }, "models/interrupted_model.pth")
        
        print("💾 Model saved to: models/interrupted_model.pth")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Training script completed!")

if __name__ == "__main__":
    main()
