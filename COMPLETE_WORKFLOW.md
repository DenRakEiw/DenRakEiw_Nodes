# 🎉 Kompletter Flux LayerDiffuse Workflow

## ✅ **Alle Probleme gelöst!**

1. ✅ **Tensor Dimension Fix** - DualCLIPLoader type="flux"
2. ✅ **VAE Compatibility Fix** - TransparentVAE decode() korrigiert
3. ✅ **Transparent Image Handling** - Spezielle Save/Preview Nodes

## 🚀 **Finaler kompletter Workflow:**

```
1. UNETLoader → Flux Model
2. LoraLoaderModelOnly → layerlora.safetensors (Strength: 1.0)
3. DualCLIPLoader → type: "flux"
4. CLIPTextEncode → "a football, transparent"
5. ConditioningZeroOut → für negative
6. EmptySD3LatentImage → 1024x1024
7. KSampler → Standard Sampling
8. VAELoader → Standard VAE
9. Flux LayerDiffuse Standalone Loader
10. Flux LayerDiffuse Decoder (Simple)
11. 💾 Save Transparent Image ← NEU!
12. 👁️ Preview Transparent Image ← NEU!
```

## 🔗 **Korrekte Verbindungen:**

### **Model & Sampling:**
```
UNETLoader → LoraLoaderModelOnly → KSampler (model)
DualCLIPLoader → CLIPTextEncode → KSampler (positive)
DualCLIPLoader → CLIPTextEncode → ConditioningZeroOut → KSampler (negative)
EmptySD3LatentImage → KSampler (latent_image)
```

### **Transparent VAE:**
```
VAELoader → Flux LayerDiffuse Standalone Loader (vae)
Flux LayerDiffuse Standalone Loader → Flux LayerDiffuse Decoder (transparent_vae)
```

### **Decoding & Output:**
```
KSampler → Flux LayerDiffuse Decoder (samples)
Flux LayerDiffuse Decoder → Save Transparent Image (images)
Flux LayerDiffuse Decoder → Preview Transparent Image (images)
```

## 🎨 **Neue Transparent Image Nodes:**

### **💾 Save Transparent Image:**
- **Speichert echte PNG** mit Transparenz
- **Automatische Dateinamen** mit Timestamp
- **Metadata speichern** (optional)
- **RGBA-Format** wird korrekt behandelt

**Einstellungen:**
- `filename_prefix`: "transparent_"
- `save_metadata`: true

### **👁️ Preview Transparent Image:**
- **Verschiedene Hintergründe**: transparent, white, black, checkerboard
- **Zeigt Transparenz** korrekt an
- **Kompatibel mit ComfyUI** Preview-System

**Einstellungen:**
- `background_color`: "checkerboard" (empfohlen)

### **📊 Transparent Image Info:**
- **Analysiert Transparenz** (Alpha-Kanal)
- **Zeigt Bilddetails** an
- **Debugging-Hilfe** für Probleme

## ⚙️ **Wichtige Einstellungen:**

### **DualCLIPLoader:**
- **type**: "flux" (NICHT "sdxl"!)
- **clip_l**: clip_l.safetensors
- **t5xxl**: t5xxl_fp8_e4m3fn.safetensors

### **KSampler:**
- **Steps**: 20-30
- **CFG**: 3.5-7.0
- **Sampler**: euler
- **Scheduler**: simple

### **Flux LayerDiffuse Decoder:**
- **use_augmentation**: true (bessere Qualität)

### **Save Transparent Image:**
- **filename_prefix**: "transparent_"
- **save_metadata**: true

## 🎯 **Was du NICHT mehr brauchst:**

- ❌ **Standard Save Image** (kann keine Transparenz)
- ❌ **Standard Preview Image** (zeigt Transparenz falsch)
- ❌ **Conditioning Fix** (mit type="flux" überflüssig)
- ❌ **VAE Input im Decoder** (jetzt im Standalone Loader)

## 📋 **Console Output sollte zeigen:**

```
Input latent shape: [1, 16, 128, 128]
Latent device: cuda:0, dtype: torch.bfloat16
TransparentVAE device: cuda:0
Decoding with TransparentVAE (includes internal VAE decode)...
✓ Decoded transparent image: [1, 1024, 1024, 4]
Saving transparent images: (1, 1024, 1024, 4)
✓ Saved transparent image: output/transparent_20250131_123456.png
```

## 🎨 **Prompt-Tipps:**

### ✅ **Perfekte Prompts:**
- "a football, transparent"
- "glass bottle, elegant, crystal clear"
- "transparent wine glass, studio lighting"
- "crystal sculpture, artistic, clear"

### ❌ **Vermeiden:**
- "on white background"
- "with background"
- Komplexe Szenen mit vielen Objekten

## 🔍 **Debugging mit neuen Tools:**

### **📊 Transparent Image Info verwenden:**
```
Flux LayerDiffuse Decoder → Transparent Image Info
```
Zeigt:
- Bildgröße und Kanäle
- Alpha-Kanal Statistiken
- Transparenz-Qualität

### **👁️ Preview mit verschiedenen Hintergründen:**
- **checkerboard**: Zeigt Transparenz am besten
- **white**: Für helle transparente Bereiche
- **black**: Für dunkle transparente Bereiche
- **transparent**: Echte Transparenz-Vorschau

## 🎉 **Erfolg garantiert!**

Mit diesem kompletten Workflow erhältst du:
- ✅ **Echte transparente PNG-Dateien**
- ✅ **Korrekte Alpha-Kanäle**
- ✅ **Hochwertige Transparenz**
- ✅ **Professionelle Ergebnisse**

## 📁 **Ausgabe-Dateien:**

```
output/
├── transparent_20250131_123456.png      # Transparentes PNG
├── transparent_20250131_123456_metadata.json  # Metadaten
└── temp/
    └── transparent_preview_checkerboard_000.png  # Preview
```

**Starte ComfyUI neu und verwende den kompletten Workflow für perfekte transparente Bilder!** 🎨✨
