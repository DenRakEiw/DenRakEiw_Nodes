# 🎨 Flux LayerDiffuse - Einfacher Workflow

## ✅ **Korrekter Workflow (funktioniert garantiert!)**

### **Schritt 1: Nodes hinzufügen**

1. **Load Diffusion Model** → Flux Modell laden
2. **Load LoRA** → layerlora.safetensors (Strength: 1.0)
3. **CLIP Text Encode** (Positive) → "glass bottle, transparent"
4. **CLIP Text Encode** (Negative) → "" (leer lassen)
5. **Empty Latent Image** → 1024x1024
6. **KSampler** → Standard ComfyUI Sampler
7. **🔧 Flux LayerDiffuse Standalone Loader** → TransparentVAE laden
8. **🔍 Flux LayerDiffuse Decoder (Simple)** → Transparenz dekodieren
9. **Save Image** → Bild speichern

### **Schritt 2: Verbindungen**

```
Load Diffusion Model → KSampler (model)
Load LoRA → (wird automatisch auf model angewendet)
CLIP Text Encode (Positive) → KSampler (positive)
CLIP Text Encode (Negative) → KSampler (negative)
Empty Latent Image → KSampler (latent_image)

KSampler (latent) → Flux LayerDiffuse Decoder (samples)
Flux LayerDiffuse Standalone Loader (transparent_vae) → Flux LayerDiffuse Decoder (transparent_vae)

Flux LayerDiffuse Decoder (transparent_image) → Save Image
```

### **Schritt 3: Einstellungen**

**KSampler:**
- Steps: 20-30
- CFG: 3.5-7.0
- Sampler: euler
- Scheduler: normal
- Denoise: 1.0

**Flux LayerDiffuse Standalone Loader:**
- transparent_vae_checkpoint: "TransparentVAE.pth"
- alpha: 300.0
- latent_channels: 16
- dtype: bfloat16

**Flux LayerDiffuse Decoder:**
- use_augmentation: true (für bessere Qualität)

## 📁 **Datei-Platzierung**

```
ComfyUI/
├── models/
│   ├── diffusion_models/    # Flux Modell hier
│   ├── loras/               # layerlora.safetensors hier
│   └── vae/                 # TransparentVAE.pth hier
```

## 💡 **Prompts für transparente Bilder**

### ✅ **Gute Prompts:**
- "glass bottle, elegant, transparent"
- "crystal wine glass, high quality"
- "transparent sculpture, artistic"
- "glass vase, studio lighting"

### ❌ **Vermeiden:**
- "on white background"
- "with background"
- Hintergrund-Beschreibungen

## 🔧 **Troubleshooting**

### **Problem: "TransparentVAE.pth not found"**
- Verwende **ℹ️ Flux LayerDiffuse Info** → "check_files"
- Datei muss in `ComfyUI/models/vae/` liegen

### **Problem: Schlechte Transparenz**
- CFG erhöhen (5.0-7.0)
- Augmentation aktivieren
- Mehr Steps verwenden (30-50)

### **Problem: Sampling-Fehler**
- **WICHTIG**: Verwende **KSampler**, nicht FluxLayerDiffuseSampler!
- Stelle sicher, dass LoRA korrekt geladen ist

### **Problem: Speicher-Fehler**
- Bildgröße reduzieren (512x512)
- dtype: float16 verwenden
- Augmentation deaktivieren

## 🎯 **Workflow-Hilfe**

Verwende **📖 Flux LayerDiffuse Workflow Helper** für:
- basic_workflow: Grundlegende Anleitung
- advanced_workflow: Erweiterte Tipps
- troubleshooting: Problemlösungen

## ⚡ **Quick Start**

1. **Starte ComfyUI neu**
2. **Verwende ℹ️ Flux LayerDiffuse Info** → "check_files"
3. **Falls Dateien fehlen** → "setup_guide" für Download-Links
4. **Folge dem obigen Workflow**
5. **Verwende 📖 Workflow Helper** bei Problemen

## 🎉 **Erfolg garantiert!**

Dieser Workflow verwendet **nur Standard ComfyUI Nodes** + unsere einfachen Decoder. 
Keine komplexen Sampling-Systeme, keine Kompatibilitätsprobleme!

**Der Schlüssel**: Verwende **KSampler** für das Sampling und unseren **Decoder** nur für die Transparenz!
