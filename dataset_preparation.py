#!/usr/bin/env python3
"""
📊 DATASET PREPARATION FOR LATENT UPSCALER
Erstellt große, hochqualitative Datasets aus verschiedenen Quellen:
- DIV2K Dataset
- Flickr2K Dataset  
- Custom Images
- VAE Encoding zu Latents
"""

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
import os
import json
import numpy as np
from PIL import Image
import requests
import zipfile
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class DatasetDownloader:
    """Automatischer Download verschiedener Datasets"""
    
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def download_file(self, url, filepath):
        """Download mit Progress Bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def download_div2k(self):
        """Download DIV2K Dataset (800 Training + 100 Validation)"""
        print("📥 Downloading DIV2K Dataset...")
        div2k_dir = os.path.join(self.base_dir, "div2k")
        os.makedirs(div2k_dir, exist_ok=True)
        
        urls = {
            "DIV2K_train_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "DIV2K_valid_HR.zip": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
        }
        
        for filename, url in urls.items():
            zip_path = os.path.join(div2k_dir, filename)
            if not os.path.exists(zip_path):
                print(f"📥 Downloading {filename}...")
                self.download_file(url, zip_path)
                
                # Extract
                print(f"📂 Extracting {filename}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(div2k_dir)
        
        print("✅ DIV2K Dataset ready!")
        return div2k_dir
    
    def download_sample_images(self):
        """Download zusätzliche Sample Images"""
        print("📥 Downloading additional sample images...")
        samples_dir = os.path.join(self.base_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        # Unsplash Sample URLs (hochqualitative Bilder)
        sample_urls = [
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=2048&q=80",
            "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=2048&q=80", 
            "https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=2048&q=80",
            "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=2048&q=80",
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=2048&q=80"
        ]
        
        for i, url in enumerate(sample_urls):
            filepath = os.path.join(samples_dir, f"sample_{i+1}.jpg")
            if not os.path.exists(filepath):
                try:
                    self.download_file(url, filepath)
                except Exception as e:
                    print(f"⚠ Failed to download sample {i+1}: {e}")
        
        print("✅ Sample images ready!")
        return samples_dir

class VAELatentEncoder:
    """Konvertiert Bilder zu VAE Latents - NUR ECHTE WAN VAE!"""

    # OFFIZIELLE WAN VAE MODELLE (KEINE FAKES!)
    OFFICIAL_WAN_VAES = {
        "../../models/vae/Wan2.2_VAE_official.safetensors": "WAN 2.2 VAE Official - EMPFOHLEN",
        "../../models/vae/Wan2.1_VAE_official.pth": "WAN 2.1 VAE Official - EMPFOHLEN",
        "../../models/vae/Wan2_1_VAE_bf16.safetensors": "WAN 2.1 VAE BF16 - EMPFOHLEN",
        "../../models/vae/wan_2.1_vae.safetensors": "WAN 2.1 VAE - EMPFOHLEN"
    }

    def __init__(self, vae_model="../../models/vae/Wan2.2_VAE_official.safetensors", device="cuda"):
        self.device = device
        self.vae_model = vae_model

        # KRITISCHE PRÜFUNG: Nur WAN VAE verwenden!
        if vae_model not in self.OFFICIAL_WAN_VAES:
            raise ValueError(f"❌ FAKE VAE ERKANNT! '{vae_model}' ist NICHT WAN VAE!\n"
                           f"✅ Verwende nur: {list(self.OFFICIAL_WAN_VAES.keys())}")

        print(f"🔧 Loading OFFICIAL WAN VAE: {vae_model}")
        print(f"✅ {self.OFFICIAL_WAN_VAES[vae_model]}")

        # Lade WAN VAE von lokalem Pfad
        self.vae = self._load_wan_vae(vae_model).to(device)
        self.vae.eval()

        # VAE-Validierung
        self._validate_vae()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])

    def _load_wan_vae(self, vae_path):
        """Lade WAN VAE von lokalem Pfad"""
        print(f"🔧 Loading WAN VAE from: {vae_path}")

        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"❌ WAN VAE nicht gefunden: {vae_path}")

        try:
            if vae_path.endswith('.safetensors'):
                # Lade Safetensors WAN VAE
                from safetensors.torch import load_file
                state_dict = load_file(vae_path)

                # Erstelle VAE-Architektur (WAN VAE kompatibel)
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_config({
                    "in_channels": 3,
                    "out_channels": 3,
                    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
                    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                    "block_out_channels": [128, 256, 512, 512],
                    "layers_per_block": 2,
                    "act_fn": "silu",
                    "latent_channels": 4,
                    "norm_num_groups": 32,
                    "sample_size": 512,
                    "scaling_factor": 0.18215
                })

                # Lade State Dict
                vae.load_state_dict(state_dict, strict=False)

            elif vae_path.endswith('.pth'):
                # Lade PyTorch WAN VAE
                vae = torch.load(vae_path, map_location='cpu')
                if hasattr(vae, 'state_dict'):
                    # Falls es ein komplettes Model-Objekt ist
                    pass
                else:
                    # Falls es nur state_dict ist
                    from diffusers import AutoencoderKL
                    vae_model = AutoencoderKL.from_config({
                        "in_channels": 3,
                        "out_channels": 3,
                        "latent_channels": 4,
                        "scaling_factor": 0.18215
                    })
                    vae_model.load_state_dict(vae)
                    vae = vae_model

            print("✅ WAN VAE successfully loaded!")
            return vae

        except Exception as e:
            raise RuntimeError(f"❌ Fehler beim Laden der WAN VAE: {e}")

    def _validate_vae(self):
        """Validiere VAE-Funktionalität"""
        print("🔍 Validating VAE functionality...")

        try:
            # Test mit Dummy-Input
            test_input = torch.randn(1, 3, 512, 512).to(self.device)

            with torch.no_grad():
                # Encode
                latent = self.vae.encode(test_input).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor

                # Prüfe Dimensionen
                expected_shape = (1, 4, 64, 64)
                if latent.shape != expected_shape:
                    raise ValueError(f"❌ FALSCHE VAE! Latent shape: {latent.shape}, erwartet: {expected_shape}")

                # Prüfe Scaling Factor
                scaling_factor = self.vae.config.scaling_factor
                if abs(scaling_factor - 0.18215) > 0.001:
                    print(f"⚠️ Ungewöhnlicher Scaling Factor: {scaling_factor} (erwartet: ~0.18215)")

                # Decode Test
                decoded = self.vae.decode(latent / self.vae.config.scaling_factor).sample

                if decoded.shape != test_input.shape:
                    raise ValueError(f"❌ VAE DECODE FEHLER! {decoded.shape} != {test_input.shape}")

                print("✅ VAE validation successful!")
                print(f"✅ Latent shape: {latent.shape}")
                print(f"✅ Scaling factor: {scaling_factor}")

        except Exception as e:
            raise RuntimeError(f"❌ VAE VALIDATION FAILED: {e}")
    
    def encode_image(self, image_path):
        """Encode einzelnes Bild zu Latent"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Encode to latent
            with torch.no_grad():
                latent = self.vae.encode(tensor).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
            
            return latent.squeeze(0).cpu()
        
        except Exception as e:
            print(f"⚠ Error encoding {image_path}: {e}")
            return None
    
    def encode_directory(self, image_dir, output_dir, max_images=None):
        """Encode alle Bilder in einem Verzeichnis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"🔄 Encoding {len(image_files)} images to latents...")
        
        successful = 0
        for i, image_path in enumerate(tqdm(image_files, desc="Encoding")):
            latent = self.encode_image(image_path)
            if latent is not None:
                # Save latent
                output_path = os.path.join(output_dir, f"latent_{i:06d}.pt")
                torch.save(latent, output_path)
                successful += 1
        
        print(f"✅ Successfully encoded {successful}/{len(image_files)} images")
        return successful

class LatentDatasetCreator:
    """Erstellt strukturierte Latent-Datasets"""
    
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        self.downloader = DatasetDownloader(base_dir)
        self.encoder = VAELatentEncoder()
    
    def create_training_dataset(self, target_size=2000):
        """Erstellt großes Training-Dataset"""
        print("🚀 Creating Large Training Dataset...")
        
        # Download datasets
        div2k_dir = self.downloader.download_div2k()
        samples_dir = self.downloader.download_sample_images()
        
        # Create output directories
        train_latents_dir = os.path.join(self.base_dir, "latents", "train")
        val_latents_dir = os.path.join(self.base_dir, "latents", "validation")
        
        # Encode DIV2K training images
        div2k_train_dir = os.path.join(div2k_dir, "DIV2K_train_HR")
        if os.path.exists(div2k_train_dir):
            print("🔄 Encoding DIV2K training images...")
            self.encoder.encode_directory(
                div2k_train_dir, 
                train_latents_dir, 
                max_images=min(800, target_size)
            )
        
        # Encode DIV2K validation images
        div2k_val_dir = os.path.join(div2k_dir, "DIV2K_valid_HR")
        if os.path.exists(div2k_val_dir):
            print("🔄 Encoding DIV2K validation images...")
            self.encoder.encode_directory(
                div2k_val_dir, 
                val_latents_dir, 
                max_images=100
            )
        
        # Encode sample images
        if os.path.exists(samples_dir):
            print("🔄 Encoding sample images...")
            self.encoder.encode_directory(
                samples_dir, 
                train_latents_dir
            )
        
        # Create dataset info
        self.create_dataset_info(train_latents_dir, val_latents_dir)
        
        print("✅ Large Training Dataset created!")
        return train_latents_dir, val_latents_dir
    
    def create_dataset_info(self, train_dir, val_dir):
        """Erstellt Dataset-Informationen"""
        info = {
            "dataset_name": "Advanced Latent Upscaler Dataset",
            "version": "2.0",
            "description": "Large-scale dataset for training latent upscalers",
            "train_samples": len([f for f in os.listdir(train_dir) if f.endswith('.pt')]),
            "val_samples": len([f for f in os.listdir(val_dir) if f.endswith('.pt')]),
            "input_size": [4, 32, 32],
            "output_size": [4, 64, 64],
            "vae_model": "stabilityai/sd-vae-ft-mse",
            "created_by": "Advanced Latent Upscaler Trainer V2.0"
        }
        
        info_path = os.path.join(self.base_dir, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"📊 Dataset Info:")
        print(f"  Training samples: {info['train_samples']}")
        print(f"  Validation samples: {info['val_samples']}")
        print(f"  Info saved to: {info_path}")

class LatentAugmentationDataset(torch.utils.data.Dataset):
    """Dataset mit erweiterten Augmentationen"""
    
    def __init__(self, latent_dir, augment=True):
        self.latent_dir = latent_dir
        self.augment = augment
        self.files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load latent
        latent_path = os.path.join(self.latent_dir, self.files[idx])
        latent = torch.load(latent_path)
        
        # Ensure correct size (should be 4x64x64)
        if latent.shape[-1] != 64:
            latent = F.interpolate(
                latent.unsqueeze(0), 
                size=(64, 64), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Create low-res input (32x32)
        low_res = F.interpolate(
            latent.unsqueeze(0), 
            size=(32, 32), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # High-res target (64x64)
        high_res = latent
        
        # Advanced augmentations
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
            
            # Slight noise injection
            if torch.rand(1) > 0.8:
                noise_scale = 0.01
                low_res += torch.randn_like(low_res) * noise_scale
        
        return low_res, high_res

def main():
    """Hauptfunktion für Dataset-Erstellung"""
    print("📊 DATASET PREPARATION FOR ADVANCED TRAINING")
    print("=" * 50)
    
    # Create dataset
    creator = LatentDatasetCreator()
    train_dir, val_dir = creator.create_training_dataset(target_size=2000)
    
    # Test dataset loading
    print("\n🧪 Testing dataset loading...")
    train_dataset = LatentAugmentationDataset(train_dir, augment=True)
    val_dataset = LatentAugmentationDataset(val_dir, augment=False)
    
    print(f"✅ Training dataset: {len(train_dataset)} samples")
    print(f"✅ Validation dataset: {len(val_dataset)} samples")
    
    # Test sample
    if len(train_dataset) > 0:
        low_res, high_res = train_dataset[0]
        print(f"📏 Sample shapes: {low_res.shape} -> {high_res.shape}")
    
    print("\n🎉 Dataset preparation completed!")
    print(f"📁 Training latents: {train_dir}")
    print(f"📁 Validation latents: {val_dir}")

if __name__ == "__main__":
    main()
