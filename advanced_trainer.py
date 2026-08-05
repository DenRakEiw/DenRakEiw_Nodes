#!/usr/bin/env python3
"""
🚀 ADVANCED LATENT UPSCALER TRAINER V2.0
Verbessertes Training-System mit:
- Perceptual Loss
- Residual Architecture  
- Data Augmentation
- Progressive Training
- Multiple Datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg19
import os
import json
import numpy as np
from PIL import Image
import requests
import zipfile
from tqdm import tqdm
import matplotlib.pyplot as plt

class PerceptualLoss(nn.Module):
    """Perceptual Loss using VGG19 features"""
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16])  # Up to conv3_4
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return nn.MSELoss()(pred_features, target_features)

class ResidualBlock(nn.Module):
    """Residual Block for better gradient flow"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual

class AdvancedLatentUpscaler(nn.Module):
    """Verbesserte Architektur mit Residual Blocks"""
    def __init__(self, input_channels=4, output_channels=4, num_residual_blocks=6):
        super().__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(input_channels, 64, 3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_residual_blocks)
        ])
        
        # Upsampling layers
        self.upsample1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),  # 64 channels -> 64 channels, 2x upscale
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, output_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial features
        features = self.initial_conv(x)
        
        # Residual processing
        for block in self.residual_blocks:
            features = block(features)
        
        # Upsampling
        upsampled = self.upsample1(features)
        
        # Final output
        output = self.final_conv(upsampled)
        
        return output

class LatentDataset(Dataset):
    """Dataset für Latent-Tensoren mit Data Augmentation"""
    def __init__(self, data_dir, transform=None, augment=True):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        
        # Lade alle .pt Dateien
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        print(f"📁 Gefunden: {len(self.files)} Latent-Dateien")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Lade Latent-Tensor
        file_path = os.path.join(self.data_dir, self.files[idx])
        latent = torch.load(file_path)
        
        # Erstelle Low-Res Version (32x32)
        low_res = torch.nn.functional.interpolate(
            latent.unsqueeze(0), 
            size=(32, 32), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # High-Res Target (64x64)
        high_res = torch.nn.functional.interpolate(
            latent.unsqueeze(0), 
            size=(64, 64), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Data Augmentation
        if self.augment and np.random.random() > 0.5:
            # Random flip
            if np.random.random() > 0.5:
                low_res = torch.flip(low_res, [2])
                high_res = torch.flip(high_res, [2])
            if np.random.random() > 0.5:
                low_res = torch.flip(low_res, [1])
                high_res = torch.flip(high_res, [1])
        
        return low_res, high_res

class DatasetDownloader:
    """Automatischer Dataset-Download"""
    
    @staticmethod
    def download_div2k(data_dir="datasets/div2k"):
        """Download DIV2K Dataset"""
        os.makedirs(data_dir, exist_ok=True)
        
        urls = {
            "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "valid_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
        }
        
        for name, url in urls.items():
            zip_path = os.path.join(data_dir, f"{name}.zip")
            if not os.path.exists(zip_path):
                print(f"📥 Downloading {name}...")
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f, tqdm(
                    desc=name,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                # Extract
                print(f"📂 Extracting {name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
        
        print("✅ DIV2K Dataset downloaded!")
        return data_dir

class AdvancedTrainer:
    """Verbesserter Trainer mit allen Features"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss().to(device)
        
        # Optimizer mit besseren Parametern
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=100, 
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def combined_loss(self, pred, target, alpha=1.0, beta=0.0):
        """Kombinierte Loss: MSE ONLY für Latent Training"""
        # Für Latent-Training verwenden wir NUR MSE Loss
        # Perceptual Loss funktioniert nicht mit 4-Channel Latents

        mse = self.mse_loss(pred, target)

        # Perceptual Loss deaktiviert für Latent-Training
        # perceptual = self.perceptual_loss(pred_rgb, target_rgb)

        return alpha * mse  # + beta * perceptual
    
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
    
    def train(self, train_loader, val_loader, epochs=100, save_dir="models"):
        """Haupttraining-Loop"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        print(f"🚀 Starting Advanced Training for {epochs} epochs...")
        
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
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  💾 New best model saved! (Val Loss: {val_loss:.6f})")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print("✅ Training completed!")
        return self.history

def main():
    """Hauptfunktion für verbessertes Training"""
    print("🚀 ADVANCED LATENT UPSCALER TRAINER V2.0")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Model
    model = AdvancedLatentUpscaler(
        input_channels=4, 
        output_channels=4, 
        num_residual_blocks=8
    )
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Datasets
    print("📊 Preparing datasets...")
    train_dataset = LatentDataset("training_data/latents", augment=True)
    val_dataset = LatentDataset("validation_data/latents", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Trainer
    trainer = AdvancedTrainer(model, device)
    
    # Training
    history = trainer.train(train_loader, val_loader, epochs=200)
    
    print("🎉 Advanced Training completed!")

if __name__ == "__main__":
    main()
