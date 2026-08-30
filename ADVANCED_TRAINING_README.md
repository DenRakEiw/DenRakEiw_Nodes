# 🚀 Advanced Latent Upscaler Training V2.0

A complete, improved training system for high-quality latent upscalers, using modern deep learning techniques.

## 🎯 Improvements over V1.0

### ✨ **Neue Features:**
- **🧠 Residual Architecture** - better gradient flow
- **👁️ Perceptual Loss** - more realistic results  
- **📊 Large datasets** - DIV2K + Flickr2K (2000+ images)
- **🔄 Data Augmentation** - Flips, Rotationen, Noise
- **📈 Progressive Training** - Cosine Annealing LR
- **📊 Monitoring** - Plots, Logs, Checkpoints
- **⚡ Auto-Setup** - one-click installation

### 🏗️ **Architecture improvements:**
- **Residual Blocks** instead of plain convolutions
- **LeakyReLU** instead of Tanh, for less smoothing
- **PixelShuffle** for better upsampling
- **Gradient Clipping** for stable training
- **AdamW Optimizer** with weight decay

## 🚀 Quick Start

### **Option 1: one-click training**
```bash
python quick_start_training.py
```

### **Option 2: Manuelles Training**
```bash
# 1. Dataset vorbereiten
python dataset_preparation.py

# 2. Training starten
python train_advanced_upscaler.py
```

## 📊 Datasets

### **Automatisch heruntergeladen:**
- **DIV2K Dataset** (800 Training + 100 Validation)
- **Sample images** from Unsplash
- **Automatische VAE-Kodierung** zu Latents

### **Supported formats:**
- JPG, PNG, BMP, TIFF
- Automatic resizing to 512x512
- VAE-Kodierung zu 4x64x64 Latents

## 🏗️ Architecture

```python
AdvancedLatentUpscaler(
    input_channels=4,      # VAE Latent Channels
    output_channels=4,     # VAE Latent Channels  
    num_residual_blocks=8  # Anzahl Residual Blocks
)
```

### **Netzwerk-Flow:**
```
Input [4x32x32] 
    ↓
Initial Conv [64 channels]
    ↓
8x Residual Blocks [64 channels]
    ↓
PixelShuffle Upsampling [2x]
    ↓
Final Conv [4 channels]
    ↓
Output [4x64x64]
```

## 🎯 Loss Function

**Kombinierte Loss:**
```python
total_loss = 0.7 * MSE_loss + 0.3 * Perceptual_loss
```

- **MSE Loss**: Pixel-genaue Rekonstruktion
- **Perceptual Loss**: VGG19-based feature similarity

## ⚙️ Training Configuration

```python
config = {
    # Model
    'num_residual_blocks': 8,
    
    # Training  
    'epochs': 200,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    
    # Loss weights
    'mse_weight': 0.7,
    'perceptual_weight': 0.3,
    
    # Dataset
    'dataset_size': 2000,
    'augmentation': True
}
```

## 📈 Monitoring

### **Automatische Plots:**
- Training/Validation Loss Curves
- Learning Rate Schedule
- Loss Gradients
- Overfitting Detection

### **Checkpoints:**
- `best_model.pth` - Bestes Validation Model
- `final_advanced_upscaler.pth` - Finales Model
- `checkpoint_epoch_X.pth` - Alle 10 Epochen

## 🔧 Hardware Requirements

### **Minimum:**
- **GPU**: 6GB VRAM (GTX 1060, RTX 2060)
- **RAM**: 16GB
- **Storage**: 10 GB for the datasets

### **Empfohlen:**
- **GPU**: 12GB+ VRAM (RTX 3080, RTX 4070)
- **RAM**: 32GB
- **Storage**: 50 GB for the large datasets

### **Batch Size Empfehlungen:**
- **6-8GB VRAM**: batch_size = 8
- **8-12GB VRAM**: batch_size = 16  
- **12GB+ VRAM**: batch_size = 32

## 📁 File layout

```
denrakeiw_nodes/
├── advanced_trainer.py          # Trainer-Klassen
├── dataset_preparation.py       # Dataset-Download & Prep
├── train_advanced_upscaler.py   # Main Training Script
├── quick_start_training.py      # one-click setup
├── wan_nn_latent_upscaler.py   # ComfyUI Node
└── datasets/                    # Auto-erstellt
    ├── div2k/                   # DIV2K Dataset
    ├── latents/                 # encoded latents
    │   ├── train/              # Training Latents
    │   └── validation/         # Validation Latents
    └── dataset_info.json       # Dataset Info
```

## 🎮 After training

### **1. Model in ComfyUI verwenden:**
```bash
# Kopiere bestes Model
cp models/best_model.pth /path/to/ComfyUI/models/upscale_models/

# Starte ComfyUI neu
```

### **2. Node verwenden:**
- Suche nach "Universal Latent Upscaler"
- Verbinde Latent Input → Node → Latent Output
- Enjoy twice the quality! 🚀

## 🔬 Erweiterte Optionen

### **Adding a custom dataset:**
```python
# Add your own images
creator = LatentDatasetCreator()
creator.encoder.encode_directory(
    "my_images/", 
    "datasets/latents/train/"
)
```

### **Training fortsetzen:**
```python
# Lade Checkpoint
checkpoint = torch.load("models/checkpoint_epoch_50.pth")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### **Hyperparameter-Tuning:**
```python
# Things to experiment with:
- num_residual_blocks: 4, 6, 8, 12
- learning_rate: 1e-5, 5e-5, 1e-4, 2e-4  
- loss_weights: (0.8, 0.2), (0.6, 0.4)
- batch_size: 8, 16, 32
```

## 🐛 Troubleshooting

### **CUDA Out of Memory:**
```python
# Reduziere batch_size
config['batch_size'] = 8

# Oder verwende Gradient Accumulation
config['accumulate_grad_batches'] = 2
```

### **Slow Training:**
```python
# Raise num_workers
config['num_workers'] = 8

# Verwende pin_memory
pin_memory=True
```

### **Poor Quality:**
```python
# Raise the perceptual loss weight
config['perceptual_weight'] = 0.5

# Mehr Residual Blocks
config['num_residual_blocks'] = 12

# Train for longer
config['epochs'] = 300
```

## 📊 Expected results

### **After 50 epochs:**
- Basic upscaling ability
- Reduzierte Artefakte

### **After 100 epochs:**
- Gute Detail-Rekonstruktion
- Stabile Farben

### **After 200 epochs:**
- High-quality results
- Sharper than standard upscaling

## 🎉 Support

Bei Fragen oder Problemen:
1. Check the logs in `logs/`
2. Look at the plots in `plots/`
3. Teste verschiedene Hyperparameter

**Good luck with the training! 🚀**
