#!/usr/bin/env python3
"""
🔥 WAN VAE LATENT UPSCALER TRAINER - DENRAKEIW SUPERHERO EDITION 🔥
Optimiertes Training-System speziell für WAN VAE (16 Channels):
- 16-Channel Input/Output
- Optimierte Architektur für WAN VAE Latents
- Advanced Loss Functions
- Progressive Training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class WanVAEResidualBlock(nn.Module):
    """Optimierter Residual Block für WAN VAE Latents (16 Channels)"""
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)  # GroupNorm für bessere Stabilität
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.1)  # Leichtes Dropout für Regularisierung
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        return self.activation(out + residual)

class WanVAEAttentionBlock(nn.Module):
    """Self-Attention Block für bessere Feature-Korrelation"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        h = self.norm(x)
        b, c, h_dim, w_dim = h.shape
        
        q = self.q(h).view(b, c, h_dim * w_dim).transpose(1, 2)
        k = self.k(h).view(b, c, h_dim * w_dim)
        v = self.v(h).view(b, c, h_dim * w_dim).transpose(1, 2)
        
        # Attention
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=2)
        
        out = torch.bmm(attn, v).transpose(1, 2).view(b, c, h_dim, w_dim)
        out = self.proj_out(out)
        
        return x + out

class AdvancedWanVAEUpscaler(nn.Module):
    """
    🔥 ADVANCED WAN VAE LATENT UPSCALER 🔥
    Speziell optimiert für WAN VAE (16 Channels)
    Input: [16, 32, 32] -> Output: [16, 64, 64]
    """
    def __init__(self, input_channels=16, output_channels=16, base_channels=64, num_residual_blocks=8):
        super().__init__()
        
        print(f"🔥 Creating Advanced WAN VAE Upscaler:")
        print(f"   Input: {input_channels} channels")
        print(f"   Output: {output_channels} channels")
        print(f"   Base channels: {base_channels}")
        print(f"   Residual blocks: {num_residual_blocks}")
        
        # Initial feature extraction
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual feature processing
        self.residual_blocks = nn.ModuleList([
            WanVAEResidualBlock(base_channels) for _ in range(num_residual_blocks)
        ])
        
        # Attention block für bessere Feature-Korrelation
        self.attention = WanVAEAttentionBlock(base_channels)
        
        # Feature refinement vor Upsampling
        self.pre_upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Upsampling mit Sub-Pixel Convolution
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),  # 2x upscale
            nn.GroupNorm(8, base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Output refinement
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.GroupNorm(4, base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, output_channels, 3, padding=1),
            nn.Tanh()  # Für VAE Latent-Bereich
        )
        
        # Skip connection für bessere Gradients
        self.skip_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Initialisierung
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Optimierte Gewichts-Initialisierung"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Skip connection
        skip = self.skip_conv(x)
        
        # Main path
        features = self.initial_conv(x)
        
        # Residual processing
        for block in self.residual_blocks:
            features = block(features)
        
        # Attention
        features = self.attention(features)
        
        # Pre-upsample refinement
        features = self.pre_upsample(features)
        
        # Upsampling
        upsampled = self.upsample(features)
        
        # Output
        output = self.output_conv(upsampled)
        
        # Combine mit Skip Connection
        result = output + skip
        
        return result

class WanVAELatentDataset(Dataset):
    """Dataset für WAN VAE Latent-Paare"""
    
    def __init__(self, latent_dir, augment=True):
        self.latent_dir = latent_dir
        self.augment = augment
        
        # Finde alle Latent-Dateien
        self.files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]
        print(f"📊 WAN VAE Dataset: {len(self.files)} samples")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Lade Latent-Paar
        file_path = os.path.join(self.latent_dir, self.files[idx])
        data = torch.load(file_path, map_location='cpu')
        
        low_res = data['low_res']    # 16x32x32
        high_res = data['high_res']  # 16x64x64
        
        # Data Augmentation
        if self.augment:
            # Random flips
            if torch.rand(1) > 0.5:
                low_res = torch.flip(low_res, [2])
                high_res = torch.flip(high_res, [2])
            if torch.rand(1) > 0.5:
                low_res = torch.flip(low_res, [1])
                high_res = torch.flip(high_res, [1])
            
            # Random rotation (90 degree steps)
            if torch.rand(1) > 0.7:
                k = torch.randint(1, 4, (1,)).item()
                low_res = torch.rot90(low_res, k, [1, 2])
                high_res = torch.rot90(high_res, k, [1, 2])
        
        return low_res, high_res

class WanVAETrainer:
    """Advanced Trainer für WAN VAE Latent Upscaler"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizer mit besseren Parametern für WAN VAE
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=2e-4,  # Etwas höhere LR für WAN VAE
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=200,
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def combined_loss(self, pred, target):
        """Optimierte Loss für WAN VAE Latents"""
        # MSE für allgemeine Rekonstruktion
        mse = self.mse_loss(pred, target)
        
        # L1 für Schärfe
        l1 = self.l1_loss(pred, target)
        
        # Kombiniere Losses
        total_loss = 0.7 * mse + 0.3 * l1
        
        return total_loss
    
    def train_epoch(self, dataloader):
        """Training für eine Epoche"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (low_res, high_res) in enumerate(pbar):
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(low_res)
            loss = self.combined_loss(pred, high_res)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validation"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for low_res, high_res in dataloader:
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)
                
                pred = self.model(low_res)
                loss = self.combined_loss(pred, high_res)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=200, save_dir="wan_vae_models"):
        """Haupttraining-Loop"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"🚀 Starting WAN VAE Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {current_lr:.8f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'history': self.history
                }, os.path.join(save_dir, 'best_wan_vae_upscaler.pth'))
                print(f"  💾 New best WAN VAE model saved! (Val Loss: {val_loss:.6f})")
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, os.path.join(save_dir, f'wan_vae_checkpoint_epoch_{epoch+1}.pth'))
        
        print("✅ WAN VAE Training completed!")
        return self.history

def main():
    """Hauptfunktion für WAN VAE Training"""
    print("🔥 WAN VAE LATENT UPSCALER TRAINER")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Model
    model = AdvancedWanVAEUpscaler(
        input_channels=16,
        output_channels=16,
        base_channels=64,
        num_residual_blocks=8
    )
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Datasets (werden erstellt wenn Dataset-Creation fertig ist)
    train_dir = "wan_vae_datasets/latents/train"
    val_dir = "wan_vae_datasets/latents/validation"
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_dataset = WanVAELatentDataset(train_dir, augment=True)
        val_dataset = WanVAELatentDataset(val_dir, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # Trainer
        trainer = WanVAETrainer(model, device)
        
        # Training
        history = trainer.train(train_loader, val_loader, epochs=200)
        
        print("🎉 WAN VAE Training completed!")
    else:
        print("⚠️ WAN VAE datasets not found!")
        print("   Run wan_vae_dataset_creator.py first to create datasets")

if __name__ == "__main__":
    main()
