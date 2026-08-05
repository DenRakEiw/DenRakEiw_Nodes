# 🚀 Advanced Latent Upscaler Training V2.0

Ein komplettes, verbessertes Training-System für hochqualitative Latent-Upscaler mit modernsten Deep Learning Techniken.

## 🎯 Verbesserungen gegenüber V1.0

### ✨ **Neue Features:**
- **🧠 Residual Architecture** - Besserer Gradient Flow
- **👁️ Perceptual Loss** - Realistischere Ergebnisse  
- **📊 Große Datasets** - DIV2K + Flickr2K (2000+ Bilder)
- **🔄 Data Augmentation** - Flips, Rotationen, Noise
- **📈 Progressive Training** - Cosine Annealing LR
- **📊 Monitoring** - Plots, Logs, Checkpoints
- **⚡ Auto-Setup** - Ein-Klick Installation

### 🏗️ **Architektur-Verbesserungen:**
- **Residual Blocks** statt einfache Convolutions
- **LeakyReLU** statt Tanh für weniger Glättung
- **PixelShuffle** für besseres Upsampling
- **Gradient Clipping** für stabiles Training
- **AdamW Optimizer** mit Weight Decay

## 🚀 Quick Start

### **Option 1: Ein-Klick Training**
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
- **Sample Images** von Unsplash
- **Automatische VAE-Kodierung** zu Latents

### **Unterstützte Formate:**
- JPG, PNG, BMP, TIFF
- Automatische Größenanpassung auf 512x512
- VAE-Kodierung zu 4x64x64 Latents

## 🏗️ Architektur

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
- **Perceptual Loss**: VGG19-basierte Feature-Ähnlichkeit

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
- **Storage**: 10GB für Datasets

### **Empfohlen:**
- **GPU**: 12GB+ VRAM (RTX 3080, RTX 4070)
- **RAM**: 32GB
- **Storage**: 50GB für große Datasets

### **Batch Size Empfehlungen:**
- **6-8GB VRAM**: batch_size = 8
- **8-12GB VRAM**: batch_size = 16  
- **12GB+ VRAM**: batch_size = 32

## 📁 Datei-Struktur

```
denrakeiw_nodes/
├── advanced_trainer.py          # Trainer-Klassen
├── dataset_preparation.py       # Dataset-Download & Prep
├── train_advanced_upscaler.py   # Main Training Script
├── quick_start_training.py      # Ein-Klick Setup
├── wan_nn_latent_upscaler.py   # ComfyUI Node
└── datasets/                    # Auto-erstellt
    ├── div2k/                   # DIV2K Dataset
    ├── latents/                 # Kodierte Latents
    │   ├── train/              # Training Latents
    │   └── validation/         # Validation Latents
    └── dataset_info.json       # Dataset Info
```

## 🎮 Nach dem Training

### **1. Model in ComfyUI verwenden:**
```bash
# Kopiere bestes Model
cp models/best_model.pth /path/to/ComfyUI/models/upscale_models/

# Starte ComfyUI neu
```

### **2. Node verwenden:**
- Suche nach "Universal Latent Upscaler"
- Verbinde Latent Input → Node → Latent Output
- Genieße 2x bessere Qualität! 🚀

## 🔬 Erweiterte Optionen

### **Custom Dataset hinzufügen:**
```python
# Eigene Bilder hinzufügen
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
# Experimentiere mit:
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
# Erhöhe num_workers
config['num_workers'] = 8

# Verwende pin_memory
pin_memory=True
```

### **Poor Quality:**
```python
# Erhöhe Perceptual Loss Weight
config['perceptual_weight'] = 0.5

# Mehr Residual Blocks
config['num_residual_blocks'] = 12

# Längeres Training
config['epochs'] = 300
```

## 📊 Erwartete Ergebnisse

### **Nach 50 Epochen:**
- Grundlegende Upscaling-Fähigkeit
- Reduzierte Artefakte

### **Nach 100 Epochen:**
- Gute Detail-Rekonstruktion
- Stabile Farben

### **Nach 200 Epochen:**
- Hochqualitative Ergebnisse
- Bessere Schärfe als Standard-Upscaling

## 🎉 Support

Bei Fragen oder Problemen:
1. Prüfe die Logs in `logs/`
2. Schaue dir die Plots in `plots/` an
3. Teste verschiedene Hyperparameter

**Viel Erfolg beim Training! 🚀**
